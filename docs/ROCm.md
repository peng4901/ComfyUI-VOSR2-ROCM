# ROCm notes

This fork targets AMD GPUs through PyTorch's ROCm build. The policy below was established on a
Radeon 780M (`gfx1103`, torch 2.11.0+rocm10.0.0, ComfyUI 0.36); the mechanisms are
architecture-independent, the kernel behaviour is not.

## Attention backends

ComfyUI's `main.py` sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on every start, which is
what makes the fused SDPA backends selectable at all. Where the AOTriton kernel images for the
running arch are missing or unsuitable, they fail in two different ways:

- **Loudly and asynchronously.** The launch fails but the error is reported at a later, unrelated
  CUDA call, so the LightningDiT/DINOv2 sites surface as `CUDA error: invalid argument` inside
  DINOv2's `proj` Linear — which makes it look like a broken matmul rather than an attention
  problem.
- **Silently.** The VAE mid block's 4-D `(b, 1, N, c)` fp32 call — which upstream's own comment
  says is written that way *to select* a fused kernel — succeeds and returns an image covered in
  a regular grid. It is invisible at 512 px and obvious at 1200 px.

`models/attention.py` owns the decision. On ROCm:

| Site | Path |
|---|---|
| DINOv2, LightningDiT | Exact-math backend, single call |
| VAE mid block | Same math, chunked over the query axis |

Pinning the small attentions changes their output by at most 1/255, i.e. within run-to-run
variation. The VAE mid block cannot use the single-call path: its token count is 22500 at 1200 px
and 262144 at 4096 px, so the score matrix would be 2 GiB and 274 GiB respectively. Chunking
computes the identical result with a fixed score-tile budget instead.

On non-ROCm devices the module returns PyTorch's normal dispatch, so CUDA behaviour is unchanged.

## Isolating the grid artifact

Same code, same input, same seed, same noise; the only variable is ComfyUI's environment
variable. 150 px → 8x (1200 px), `tile_size 512`, `tile_overlap 128`, `align none`, seed 666:

| Run | Mean difference vs the clean run | 8-px cell spread (lower is cleaner) |
|---|---|---|
| Bare script, env var unset | — | 56.58 |
| Bare script, env var set, pre-fix | 20.0/255 | 72.53 |
| ComfyUI process, pre-fix | 20.0/255 | 72.43 |
| ComfyUI process, after the fix | 0.06/255 | 56.72 |

Also ruled out along the way: the input decode (ComfyUI's `LoadImage` and a plain PIL open of the
same file agree bit-for-bit), dtype (`fp16` and `fp32` differ by 0.15/255 in-process), and Sage
Attention (it does not patch `torch.nn.functional.scaled_dot_product_attention`, so the node's
calls are unaffected by `--use-sage-attention`).

## Reference contract

The upstream script is the specification. Three details differ from the port and are now explicit
inputs (`models/resize.py`, `inference.py`):

| Contract | Upstream | This fork |
|---|---|---|
| Pre-scale | `Image.BICUBIC` on the 8-bit image | Same: Pillow on the same 8-bit data. PyTorch's own bicubic uses a different kernel (`a = -0.75` against Pillow's `-0.5`) and routes through no 8-bit intermediate; on the reference's own input the two differ by up to 7.2/255, which this one-step model amplifies ~10x |
| Latent noise | `torch.randn_like(latent)` from the global CUDA RNG, seeded once at startup | `noise_mode=reference`: the same draw, with the global CPU/CUDA states saved and restored around it. A local generator produces an unrelated field (max 3.11 apart at seed 42) |
| VAE tiling | Single pass unless `--vae_tile_size` is set | `vae_tiling=auto`: single pass up to 2048 px / 4.2 MP, tiled above only to avoid OOM |

## Precision

The upstream implementation is fp32 end to end (`vae.to(device)`, `model.to(device)`, no
autocast). Deviation from an fp32 reference, matched policies, 128 px → 4x:

| `dtype` | DiT velocity mean | Final image max | Final image mean |
|---|---|---|---|
| `fp32` | 1e-5 | 1/255 | 0.50/255 |
| `fp16` | 0.0017 | 32/255 | 0.59/255 |
| `bf16` | 0.015 | 74/255 | 1.79/255 |

`default` follows ComfyUI's device policy and is fp16 on this hardware. bf16 is an order of
magnitude worse than fp16 here; a one-step model has no room to absorb that.

## Verifying a change

The reference is the stock upstream `inference_vosr_onestep.py`, run fp32 on the same machine
with the intermediates captured; the fork replays that capture stage by stage (pre-scale, latent,
noise, DINOv2 layer 17, DiT velocity, SR latent, decoded pixels). Under fp32 with the contract
above, the pre-scaled input, latent and noise are bit-identical and the final image is within
1/255, which is this hardware's floor — two identical reference runs also differ by 1/255.

Any change to `models/attention.py`, `models/resize.py`, `inference.py` or the loader's dtype
handling should be re-checked that way rather than by eye: a one-step model turns a numeric
difference that looks harmless in one stage into a visible one at the end.
