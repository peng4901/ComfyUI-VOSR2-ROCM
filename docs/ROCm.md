# ROCm notes (fork branch `rocm`)

Upstream's CI only covers Linux/CPU registration plus a CUDA lane, so nothing here was
ever exercised on AMD. This branch carries the platform policy this box needs
(AMD Radeon 780M, `gfx1103`, torch 2.11.0+rocm10.0.0, ComfyUI 0.36) and the changes
required to reproduce the reference implementation's numbers.

## 1. Attention backends

ComfyUI's `main.py` sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on every start. On
`gfx1103` the ROCm AOTriton kernel images are missing or unsuitable, so **both** the flash
and the mem-efficient SDPA backends launch and fail. The failure is reported
asynchronously at the next CUDA call, which is why it surfaces as `hipErrorInvalidValue`
inside DINOv2's `proj` Linear rather than at the attention itself.

Measured on this box with the stock upstream `inference_vosr_onestep.py`
(64x64 -> 2x, fp32, seed 42, untiled, `--align_method nofix`):

| run | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | result |
|---|---|---|
| stock upstream | unset | OK |
| stock upstream | `1` (what ComfyUI sets) | `CUDA error: invalid argument` |
| stock upstream + `SDPBackend.MATH` pin | `1` | OK |

Pinning to MATH changes the output by at most 1/255, i.e. within this box's run-to-run
variation. `models/attention.py` applies that pin to the three SDPA call sites (DINOv2 plus
the two LightningDiT attentions). The fp32 VAE is deliberately left on the default
backends: its tiles reach ~16k tokens where MATH's O(N^2) attention matrix would be a real
memory cost, and fp32 does not select the broken kernels.

## 2. Matching the reference

VOSR 2.0 is a *one-step* model: a single DiT call turns `(lq_latent, noise)` into the SR
latent, so anything that perturbs the LQ, the noise, or the arithmetic is amplified roughly
tenfold at the output. Three implementation choices did not match upstream's contract and
are now explicit, documented policies:

| contract | upstream (`inference_vosr_onestep.py`) | this branch |
|---|---|---|
| pre-scale | `raw_img.resize((w*u, h*u), Image.BICUBIC)` on the 8-bit image | `models/resize.py`: Pillow, on the same 8-bit data |
| latent noise | `torch.randn_like(lq_latent)` -- the global CUDA RNG after `manual_seed(seed)` at startup | `noise_mode=reference`: the same draw, with the global CPU/CUDA states saved and restored around it |
| VAE tiling | single-pass unless `--vae_tile_size > 0` | `vae_tiling=auto`: single-pass up to 2048px / 4.2MP, Gaussian-blended tiles above |

`F.interpolate(mode="bicubic")` is not a substitute for Pillow's resize: PyTorch's kernel
uses `a = -0.75` against Pillow's `a = -0.5`, and Pillow routes 8-bit data through an 8-bit
intermediate. Measured on the reference's own input at 2x, the two differ by up to 7.2/255
with only 51% of pixels equal -- small in the LQ, ~10% of full scale at the output. A local
`torch.Generator` is likewise not a substitute for the reference's global draw: the two
noise fields differ by up to 3.11 at seed 42, i.e. an unrelated SR result.

With all three aligned and fp32, `tools/parity_check.py` reproduces the reference exactly
through the pre-scale, the VAE latent and the noise, and lands within 1/255 on the final
PNG (`lq`, `lq_latent`, `noise` bit-identical; DiT `u` max 1.3e-4). The 1/255 residue is
this box's noise floor, not the pack: two identical *stock* reference runs also differ by
1/255.

## 3. Precision

The reference is fp32 end to end (`vae.to(device)`, `model.to(device)`, no autocast). On
this box ComfyUI's `unet_dtype()` selects **fp16** for `gfx1103`, so `dtype=default` is
fp16; `vae_dtype()` already returns fp32, which is what the Qwen VAE needs.

Deviation from the fp32 reference, matched policies, `align=none`, seed 42:

| dtype | DiT `u` mean | final max | final mean |
|---|---|---|---|
| fp32 | 1e-5 | **1/255** | 0.50/255 |
| fp16 (`= default`) | 0.0017 | 32/255 | 0.59/255 |
| bf16 | 0.015 | 74/255 | 1.79/255 |

That is the 128x128 -> 4x capture (512x512 output). The same run at 128x128 -> 2x gives
fp16 9/255 and bf16 94/255, so the max is a single-pixel statistic -- the mean (under
0.6/255 for fp16) is the stable one.

Use `fp32` to reproduce reference results. **Avoid bf16 for VOSR 2.0**: a ~0.4% relative
error is more than a one-step model can absorb, and it is an order of magnitude worse than
fp16 here.

## 4. Verification

The reference harness lives outside this repo, in `E:\ComfyUI_JZ\_vosr_diag`:

- `a_policy.py` - ROCm policy shared by the harness (SDPA pin, torch.compile off,
  MIOpen cache redirected into the workspace, DINOv2 served from local safetensors).
- `a_boot.py` - runs the stock upstream CLI unchanged under that policy.
- `a_dump.py` - the same math with every intermediate captured, plus the reference noise.
- `micro_probe.py` - the two contract probes above (noise draw, pre-scale kernels).

`outDump\intermediates.pt` (64x64 -> 2x, seed 42, fp32, untiled, `nofix`) is the parity
target: `lq`, `lq_latent`, `venc_layer17`, `noise`, DiT `u`, `sr_latent`, `sr_tensor`, final
pixels. The four stock/dump runs agree to max 1/255, so the harness is faithful.

`tools/parity_check.py` (run with ComfyUI's own interpreter) replays that dump against this
branch stage by stage and per dtype, and also checks the node contract -- every declared
input must be an `execute` parameter in the same order, since the frontend maps a saved
workflow's positional `widgets_values` onto that order.
