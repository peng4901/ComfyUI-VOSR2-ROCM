# ComfyUI-VOSR2-ROCM

An AMD ROCm build of the [ComfyUI-VOSR2](https://github.com/ylchen333/ComfyUI-VOSR2) node for
**VOSR 2.0** — one-step, 1.4B image super-resolution (LightningDiT + Qwen-Image 2D VAE +
DINOv2-L).

Upstream is developed and CI-tested on CPU and CUDA. On ROCm the node did not work out of the
box: it crashed inside the vision encoder, and once running it produced a regular grid pattern
over the upscaled image. This fork fixes both, and reproduces the upstream reference
implementation to within this hardware's 1/255 run-to-run noise floor.

- Upstream node: <https://github.com/ylchen333/ComfyUI-VOSR2>
- Official VOSR code: <https://github.com/cswry/VOSR>
- Official weights: <https://huggingface.co/CSWRY/VOSR>

## What this fork changes

### 1. Attention backends

ComfyUI's `main.py` sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on every start, which is
what makes PyTorch's fused SDPA backends selectable at all. On `gfx1103` the AOTriton kernel
images are missing or unsuitable, and they fail in two different ways:

| Call site | Failure mode |
|---|---|
| LightningDiT and DINOv2 | Launches, then reports asynchronously at the next CUDA call — surfaces as `CUDA error: invalid argument` inside DINOv2's `proj` Linear |
| VAE mid block (4-D `(b, 1, N, c)`, fp32) | Does not fail. Returns an image covered in a regular grid — invisible at 512 px, obvious at 1200 px |

`models/attention.py` is the single place that decides backend policy. It pins the exact-math
backend for the small attentions and computes the same math chunked over the query axis for the
VAE mid block, so nothing in the pipeline uses a fused ROCm kernel. Non-ROCm devices keep
PyTorch's normal dispatch, so CUDA output is unchanged.

### 2. Reference reproducibility

VOSR 2.0 is a one-step model: a small difference in the low-quality input, the latent noise or
the arithmetic becomes a large difference in the image. Three details of the ComfyUI port
diverged from the upstream inference script and are now explicit options:

| Contract | Upstream | This fork |
|---|---|---|
| Pre-scale | `Image.BICUBIC` on the 8-bit image | Same — `models/resize.py` calls Pillow on the same 8-bit data |
| Latent noise | `torch.randn_like(latent)` from the global CUDA RNG after `manual_seed(seed)` | `noise_mode=reference` (default): the same draw, with the global CPU/CUDA states restored immediately afterwards so no other node is affected |
| VAE tiling | Single pass unless `--vae_tile_size` is set | `vae_tiling=auto` (default): single pass up to 2048 px / 4.2 MP, tiled above only to avoid OOM |

With fp32 and these policies, the reference's pre-scaled input, latent and latent noise
reproduce bit-for-bit and the final image lands within **1/255** — the same difference two
identical reference runs show on this hardware.

### 3. Precision

The upstream implementation is fp32 end to end. `fp32` is now selectable, and the reduced
precision paths have been measured against the fp32 reference (128 px → 4x, 512² output):

| `dtype` | Final image vs fp32 reference |
|---|---|
| `fp32` | 1/255 max, 0.50/255 mean |
| `fp16` | 32/255 worst pixel, 0.59/255 mean |
| `bf16` | 94/255 worst pixel, up to 5.73/255 mean — not recommended for this model |

`default` follows ComfyUI's device policy (`unet_dtype()`); on this hardware that is fp16. Use
`fp32` when you need to reproduce reference results.

## Requirements

- ComfyUI with an AMD GPU and a ROCm PyTorch build.
- `einops`, `safetensors` and `huggingface_hub`, all bundled with ComfyUI.
- Tested on: AMD Radeon 780M (`gfx1103`), torch 2.11.0+rocm10.0.0, ComfyUI 0.36.
- Model files: ~7 GB total, downloaded on first use (see below).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-user>/ComfyUI-VOSR2-ROCM
# restart ComfyUI
```

The first run downloads the VOSR 2.0 bundle from
[`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR) into `ComfyUI/models/vosr2/`. When managing
the install by hand, prefer a symlink or junction from `custom_nodes/ComfyUI-VOSR2-ROCM` to a
working checkout: edits then take effect on the next ComfyUI restart without reinstalling.

## Nodes

### VOSR 2.0 Model Loader (`VOSR2ModelLoader`)

| Input | Default | Notes |
|---|---|---|
| `model` | `VOSR2` | Bundle folder under `models/vosr2/` — the DiT plus its matched VAE and vision encoder |
| `dtype` | `default` | `default` / `fp16` / `bf16` / `fp32` for the DiT and vision encoder. The VAE always runs in fp32 |

VOSR 2.0 is a fixed triple: the DiT only works with the Qwen 2D VAE it was trained against, and
its conditioning is a specific DINOv2-L layer. The VAE and vision encoder are therefore bundle
internals, not separate inputs.

### VOSR 2.0 Upscale (`VOSR2Upscale`)

| Input | Default | Range | Notes |
|---|---:|---|---|
| `model` | — | `VOSR2_MODEL` | From the loader |
| `image` | — | `IMAGE` | Single image or batch |
| `upscale` | `4` | ≥ 1, uncapped | Exact output multiplier; ~4x is the documented sweet spot |
| `seed` | `42` | ≥ 0 | Latent-noise seed (see `noise_mode`) |
| `color_alignment` | `wavelet` | `wavelet` / `adain` / `none` | Post-process against the bicubic target |
| `tile_size` | `0` | `0`–`4096`, step 64 | DiT pixel tile; `0` disables tiling |
| `tile_overlap` | `32` | `0`–`512`, step 8 | DiT tile overlap |
| `vae_tile_size` | `0` | `0`–`8192`, step 64 | VAE pixel tile used when `vae_tiling` tiles; `0` means 1024 |
| `vae_tile_overlap` | `32` | `0`–`512`, step 8 | VAE tile overlap |
| `noise_mode` | `reference` | `reference` / `isolated` | `reference` reproduces upstream's draw; `isolated` uses a private generator and leaves global RNG state alone (batch item *i* gets `seed + i`) |
| `vae_tiling` | `auto` | `auto` / `full` / `tiled` | `auto` keeps the VAE single-pass (the reference behaviour, no tile blending) up to 2048 px / 4.2 MP; `tiled` always tiles at `vae_tile_size` |

**Tiling is not optional above 512 px.** VOSR 2.0 was trained at up to 512 px, so whenever the
*upscaled* output exceeds 512×512 set `tile_size` (e.g. `512`), or quality degrades.

## Model files

A bundle lives at `ComfyUI/models/vosr2/<bundle>/`:

```text
models/vosr2/VOSR2/
    args.json
    checkpoints/ema_model.safetensors
    Qwen-Image-vae-2d/{config.json, diffusion_pytorch_model.safetensors}
    dinov2_vitl14.safetensors
```

The loader fetches anything missing from `CSWRY/VOSR` on first use and validates `args.json`
against the fixed VOSR 2.0 architecture before constructing anything. For offline installs:

| File in `CSWRY/VOSR` | Put it at |
|---|---|
| [`VOSR2/args.json`](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/args.json) | `models/vosr2/VOSR2/args.json` |
| [`VOSR2/checkpoints/ema_model.safetensors`](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/checkpoints/ema_model.safetensors) | `models/vosr2/VOSR2/checkpoints/ema_model.safetensors` |
| [`Qwen-Image-vae-2d/config.json`](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/config.json) | `models/vosr2/VOSR2/Qwen-Image-vae-2d/config.json` |
| [`Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors`](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors) | `models/vosr2/VOSR2/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors` |
| [`torch_cache/checkpoints/dinov2_vitl14_pretrain.pth`](https://huggingface.co/CSWRY/VOSR/resolve/main/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth) | Convert to `models/vosr2/VOSR2/dinov2_vitl14.safetensors` (the loader does this automatically when downloading) |

## Memory and tiling

- `tile_size` bounds the DiT's peak memory; `vae_tiling` bounds the VAE's. They are independent.
- On ROCm every attention runs the exact-math path, whose score matrix grows with the square of
  the token count. That is why `tile_size` matters more here than on CUDA: an untiled 1200 px
  image would need ~48 GiB of attention scratch, so DiT tiling is mandatory, not an
  optimisation.
- The VAE is purely convolutional and `vae_tiling=auto` keeps it single-pass at normal sizes,
  which is both faster and closer to the reference than blending tiles. See
  [`docs/avoiding-oom.md`](docs/avoiding-oom.md).

## Verification

Everything above was measured, not assumed. The reference is the stock upstream
`inference_vosr_onestep.py` run fp32 on the same machine with every intermediate captured; the
fork replays it stage by stage:

| Stage | fp32, matched policies |
|---|---|
| Pre-scaled input, latent, latent noise | bit-identical |
| DINOv2 layer-17 conditioning | 1.2e-4 max |
| DiT velocity, SR latent | 1.3e-4 max |
| Final image, untiled 512² and 1024² | 1/255 max, 0.50/255 mean |
| Final image, tiled 8x (3x3 blend grid) | 1/255 max, 0.50/255 mean |

See [`docs/ROCm.md`](docs/ROCm.md) for the isolation experiments behind the attention fix.

## Licensing

The node code is Apache-2.0 (see [`LICENSE`](LICENSE)). Model weights are distributed by
[`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR) under their own terms; the DINOv2-L weights are
CC-BY-NC 4.0 (non-commercial). Review those terms before any commercial use.

## Credits

- **VOSR / VOSR 2.0** — Rongyuan Wu et al. ([cswry/VOSR](https://github.com/cswry/VOSR))
- **ComfyUI node** — [ylchen333/ComfyUI-VOSR2](https://github.com/ylchen333/ComfyUI-VOSR2)
- **DINOv2** — Meta AI ([facebookresearch/dinov2](https://github.com/facebookresearch/dinov2))
