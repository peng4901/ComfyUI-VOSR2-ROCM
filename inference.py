"""VOSR2 inference contract: bicubic pre-scale, pad, encode, one-step denoise,
decode, color-align, crop -- with optional DiT/VAE tiling. See VOSR2.md
"Inference contract" and "Tiling".

`forward_flexible` (models/lightningdit.py) asserts a square input -- true of
every upstream tiled call (tiles are always cropped `lt_size x lt_size`), but
not of a whole non-square image's latent. To support arbitrary aspect ratios
without tiling, the untiled path additionally pads to a square before the DiT
call and crops back afterward; this is invisible in the output (still cropped
to the exact requested size) and only engages when tile_size == 0.
"""
import logging

import torch
import torch.nn.functional as F

import comfy.utils

from .color import apply_color_alignment
from .models.resize import prescale_bicubic
from .tiled_vae import _gaussian_weights, _make_tile_grid

AE_FACTOR = 8
DIT_PATCH_SIZE = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH_SIZE  # 16

NOISE_MODES = ("reference", "isolated")
VAE_TILING_MODES = ("auto", "full", "tiled")
# `auto` keeps the reference's single-pass VAE -- no Gaussian blending between tiles --
# everywhere the image fits comfortably: VOSR 2.0's documented 4x sweet spot is
# 512 -> 2048, and the Qwen 2D VAE is purely convolutional (`attn_scales: []`), so its
# memory is linear in pixels rather than quadratic in latent tokens.
VAE_FULL_MAX_SIDE = 2048
VAE_FULL_MAX_PIXELS = VAE_FULL_MAX_SIDE * VAE_FULL_MAX_SIDE
VAE_TILE_SIZE_FALLBACK = 1024


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _pad_to_square(x: torch.Tensor):
    _, _, h, w = x.shape
    side = max(h, w)
    pad_h, pad_w = side - h, side - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _generate_noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """One item's worth of noise per call; a local CPU Generator keeps global RNG state untouched."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _reference_noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """The reference's draw, with the global RNG left exactly as it was found.

    Upstream seeds once at startup (`torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`)
    and then takes the latent noise from the *global* generator with
    `z = torch.randn_like(lq_latent)`. Reproducing a reference image therefore requires
    drawing from the same generator -- a local `Generator` yields a different field
    (max |diff| 3.11 at seed 42).

    Verified bit-for-bit against the recorded reference noise. The global CPU and CUDA
    states are saved and restored around the draw, so nothing else in a ComfyUI run is
    perturbed by this node -- without that, seeding the global RNG here would silently
    change every other node's sampler.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return torch.randn(shape, device=device, dtype=dtype)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _noise_for_batch(batch_shape, seed: int, device, dtype, noise_mode: str) -> torch.Tensor:
    """Latent noise for a whole batch, drawn the way `noise_mode` specifies."""
    if noise_mode == "reference":
        # One draw over the full batch, exactly as `torch.randn_like(lq_latent)` would.
        return _reference_noise(tuple(batch_shape), seed, device, dtype)
    return torch.stack(
        [_generate_noise(batch_shape[1:], seed + i, device, dtype) for i in range(batch_shape[0])], dim=0
    )


def _tile_params(tile_size: int, tile_overlap: int, lh: int, lw: int):
    """Pixel tile size/overlap -> latent-space, aligned to the DiT patch size."""
    lt_size = max((tile_size // AE_FACTOR // DIT_PATCH_SIZE) * DIT_PATCH_SIZE, DIT_PATCH_SIZE)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def _resize_to_target(images_bhwc01: torch.Tensor, upscale: int, device) -> torch.Tensor:
    """Reference pre-scale (Pillow BICUBIC on the 8-bit image) -> BCHW [0, 1]. See models/resize.py."""
    return prescale_bicubic(images_bhwc01, upscale, device)


def _resolve_vae_tiling(vae_tiling: str, vae_tile_size: int, vae_tile_overlap: int, h: int, w: int):
    """Decide the tile size the VAE paths get: 0 means single-pass (the reference behaviour).

    Upstream tiles the VAE only as a memory escape hatch, and its Gaussian blend is an
    approximation the reference does not otherwise pay for: `vae_tile_size > 0` used to
    activate it at any size, which silently cost quality at resolutions the full-image VAE
    handles fine. `auto` keeps the reference's single-pass VAE up to the documented 4x
    sweet spot and falls back to tiling (never OOM) above it.
    """
    tile_size = vae_tile_size if vae_tile_size > 0 else VAE_TILE_SIZE_FALLBACK
    if vae_tiling == "full":
        return 0, vae_tile_overlap
    if vae_tiling == "tiled":
        return tile_size, vae_tile_overlap
    if vae_tiling != "auto":
        raise ValueError(f"VOSR2Upscale: unknown vae_tiling mode {vae_tiling!r}.")
    if max(h, w) <= VAE_FULL_MAX_SIDE and h * w <= VAE_FULL_MAX_PIXELS:
        return 0, vae_tile_overlap
    return tile_size, vae_tile_overlap


def _run_untiled_batch(model, resized01: torch.Tensor, seed: int, vae_tile_size: int, vae_tile_overlap: int,
                       noise_mode: str) -> torch.Tensor:
    b, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded01 = _pad_to_square(padded01)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    venc_fea = model.vision_features(padded01)

    noise = _noise_for_batch(lq_latent.shape, seed, lq_latent.device, lq_latent.dtype, noise_mode)
    sr_latent = model.denoise_one_step(lq_latent, noise, venc_fea)
    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)

    return decoded_pm1[:, :, :h, :w]


def _count_dit_tiles(h: int, w: int, tile_size: int, tile_overlap: int) -> int:
    """Tiles a `run_vosr2` will submit to the DiT for one already-upscaled HxW image, for progress reporting."""
    padded_h = h + (-h) % PAD_MULTIPLE
    padded_w = w + (-w) % PAD_MULTIPLE
    lh, lw = padded_h // AE_FACTOR, padded_w // AE_FACTOR
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)
    if lh <= lt_size and lw <= lt_size:
        return 1
    return len(_make_tile_grid(lh, lt_size, lt_overlap)) * len(_make_tile_grid(lw, lt_size, lt_overlap))


def _run_tiled_single(model, resized01: torch.Tensor, seed: int, tile_size: int, tile_overlap: int,
                       vae_tile_size: int, vae_tile_overlap: int, noise_mode: str,
                       pbar: comfy.utils.ProgressBar = None) -> torch.Tensor:
    """Latent-space tiled DiT inference for one image (B=1). Ported from VOSR's tiled_latent_inference."""
    _, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    _, lc, lh, lw = lq_latent.shape
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        lq_sq = _pad_to_square(lq_latent)
        venc_fea = model.vision_features(padded01)
        noise = _noise_for_batch(lq_sq.shape, seed, lq_sq.device, lq_sq.dtype, noise_mode)
        sr_latent = model.denoise_one_step(lq_sq, noise, venc_fea)[:, :, :lh, :lw]
        if pbar is not None:
            pbar.update_absolute(pbar.current + 1)
    else:
        h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
        w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
        g_weight = _gaussian_weights(lt_size, lt_size, lc, lq_latent.device)

        tile_venc = {}
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt_size) * AE_FACTOR, padded01.shape[2])
                pw_e = min((wi + lt_size) * AE_FACTOR, padded01.shape[3])
                tile_venc[(hi, wi)] = model.vision_features(padded01[:, :, ph_s:ph_e, pw_s:pw_e])

        noise = _noise_for_batch(lq_latent.shape, seed, lq_latent.device, lq_latent.dtype, noise_mode)
        z = noise

        u_acc = torch.zeros_like(lq_latent)
        w_acc = torch.zeros_like(lq_latent)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt_size, wi + lt_size
                inp = torch.cat([lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], dim=1)
                u_tile = model.dit_velocity(inp, 1.0, 0.0, tile_venc[(hi, wi)])
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
                if pbar is not None:
                    pbar.update_absolute(pbar.current + 1)

        sr_latent = z - u_acc / w_acc

    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decoded_pm1[:, :, :h, :w]


def run_vosr2(
    model,
    images_bhwc01: torch.Tensor,
    upscale: int,
    seed: int,
    color_alignment: str,
    tile_size: int,
    tile_overlap: int,
    vae_tile_size: int,
    vae_tile_overlap: int,
    noise_mode: str = "reference",
    vae_tiling: str = "auto",
) -> torch.Tensor:
    if images_bhwc01.shape[-1] != 3:
        raise ValueError(f"VOSR2Upscale expects a 3-channel RGB IMAGE, got {images_bhwc01.shape[-1]} channels.")
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"VOSR2Upscale: unknown noise_mode {noise_mode!r}; expected one of {NOISE_MODES}.")
    if vae_tiling not in VAE_TILING_MODES:
        raise ValueError(f"VOSR2Upscale: unknown vae_tiling {vae_tiling!r}; expected one of {VAE_TILING_MODES}.")
    if tile_size > 0 and tile_overlap >= tile_size:
        raise ValueError(f"VOSR2Upscale: tile_overlap ({tile_overlap}) must be smaller than tile_size ({tile_size}).")
    if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
        raise ValueError(f"VOSR2Upscale: vae_tile_overlap ({vae_tile_overlap}) must be smaller than vae_tile_size ({vae_tile_size}).")

    device = model.dit_patcher.load_device
    resized01 = _resize_to_target(images_bhwc01, upscale, device)
    b, _, h, w = resized01.shape

    if tile_size == 0 and (h > 512 or w > 512):
        logging.warning(
            f"VOSR2Upscale: target size {w}x{h} (after {upscale}x upscale) exceeds VOSR 2.0's "
            f"native 512px training resolution with tile_size=0 (DiT tiling disabled). VOSR2.md "
            f"states tiling is not optional above 512px -- set tile_size > 0 (e.g. 512) or "
            f"quality will likely degrade."
        )
    vae_tile_size, vae_tile_overlap = _resolve_vae_tiling(vae_tiling, vae_tile_size, vae_tile_overlap, h, w)
    logging.info(
        f"VOSR2Upscale: {w}x{h} from {upscale}x, seed {seed}, noise_mode={noise_mode}, "
        f"DiT tile {tile_size or 'off'}, VAE {'tiled @%d' % vae_tile_size if vae_tile_size else 'single-pass'} "
        f"(vae_tiling={vae_tiling}), align={color_alignment}."
    )

    if tile_size > 0:
        _, _, h, w = resized01.shape
        total_tiles = b * _count_dit_tiles(h, w, tile_size, tile_overlap)
        pbar = comfy.utils.ProgressBar(total_tiles)
        # Tiling bounds peak VRAM per item, but a batch's items were still
        # accumulating on GPU here until the final cat -- peak VRAM grew with
        # batch_size x output resolution regardless of tile_size, defeating
        # the point for large multi-item batches. Move each item off the GPU
        # as soon as it's decoded so only one item's output is ever GPU-
        # resident at a time.
        #
        # Color alignment also has to happen per item here, not on the
        # concatenated batch afterward: wavelet/adain run a Gaussian blur
        # over the whole tensor at once (color.py), so doing it post-concat
        # means one CPU allocation sized to the *entire* batch's decoded
        # output -- for large images/batches that's tens of GB of system
        # RAM even though nothing is GPU-resident anymore. Aligning each
        # item right after it's decoded caps that allocation at one item's
        # size, matching the per-item VRAM bound above.
        aligned_items = []
        for i in range(b):
            item_pm1 = _run_tiled_single(
                model, resized01[i:i + 1], seed + i, tile_size, tile_overlap,
                vae_tile_size, vae_tile_overlap, noise_mode, pbar=pbar,
            ).cpu()
            item01 = (item_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
            aligned_items.append(apply_color_alignment(item01, resized01[i:i + 1].cpu(), color_alignment))
        aligned01 = torch.cat(aligned_items, dim=0)
    else:
        outputs_pm1 = _run_untiled_batch(model, resized01, seed, vae_tile_size, vae_tile_overlap, noise_mode)
        decoded01 = (outputs_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
        aligned01 = apply_color_alignment(decoded01, resized01, color_alignment)

    return aligned01.movedim(1, -1)
