# ROCm notes (fork branch `rocm`)

Upstream's CI only covers Linux/CPU registration plus a CUDA lane, so nothing here was
ever exercised on AMD. This branch carries the platform policy this box needs
(AMD Radeon 780M, `gfx1103`, torch 2.11.0+rocm10.0.0, ComfyUI 0.36).

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

## 2. Known gaps on this branch

- `dtype=default` resolves to bf16 here (the launcher passes `--bf16-unet`), while the
  reference implementation is fp32 end to end. Neither upstream nor this branch has a
  measured bf16-vs-fp32 comparison for VOSR2.
- **Noise parity is impossible today.** The reference draws its latent noise from the
  global CUDA RNG after `torch.manual_seed(seed)`; this pack uses a local CPU generator
  (`inference._generate_noise`). At seed 42 the two noise fields differ by up to 3.11, so
  no reference image can be reproduced until a `noise_mode: reference` switch exists.
- VAE tiling is reference-divergent: upstream tiles the VAE only for extreme (>~8K) sizes,
  while this pack activates it whenever `vae_tile_size > 0`.
- Process-dependent numerics (unresolved): identical code, parameters and seed produced two
  different images -- mean RGB `150.9/131.0/133.2` (std 75.1) from any ComfyUI process
  versus `139.1/114.7/113.9` (std 58.4) from a bare script calling `run_vosr2`, mean|d|
  21.19. Ruled out so far: comfy-aimdo/DynamicVRAM, XB_ToolBox's ROCm matmul tuning,
  `import nodes`, `torch.inference_mode()`, `OCL_SET_SVM_SIZE`, allocator churn,
  `--disable-comfy-compiler`, bf16-vs-fp16, and the input-image decode.

## 3. Verification

The reference harness lives outside this repo, in `E:\ComfyUI_JZ\_vosr_diag`:

- `a_policy.py` - ROCm policy shared by the harness (SDPA pin, torch.compile off,
  MIOpen cache redirected into the workspace, DINOv2 served from local safetensors).
- `a_boot.py` - runs the stock upstream CLI unchanged under that policy.
- `a_dump.py` - same math with every intermediate captured, plus the reference noise.

`outDump\intermediates.pt` (64x64 -> 2x, seed 42, fp32, untiled, `nofix`) is the parity
target for this pack: `lq`, `lq_latent`, `venc_layer17`, `noise`, DiT `u`, `sr_latent`,
`sr_tensor`, final pixels. The four stock/dump runs agree to max 1/255, so the harness is
faithful.
