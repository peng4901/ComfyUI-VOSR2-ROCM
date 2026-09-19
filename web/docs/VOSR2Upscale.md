# VOSR 2.0 Upscale

One-step super-resolution on an `IMAGE` batch using a `VOSR2_MODEL` from the
**VOSR 2.0 Model Loader**.

## Inputs

- **model** — A `VOSR2_MODEL` from the loader.
- **image** — A single image or batch, RGB (3-channel).
- **upscale** — Exact output multiplier. VOSR 2.0 was trained on degradations up
  to 16x, but ~4x is the documented sweet spot; above that, quality depends on
  the image and the target resolution rather than on the factor.
- **seed** — Latent-noise seed. With `noise_mode` = `reference` the whole batch
  is a single draw of seed `seed`, which is what the upstream script does and
  what makes its published results reproducible. With `isolated`, batch item *i*
  uses `seed + i` and results stay stable per-image as batch size changes.
- **color_alignment** — Post-process the model output against the bicubic
  target: `wavelet` (default), `adain`, or `none`.
- **tile_size** / **tile_overlap** — DiT pixel tile size and overlap. `0`
  disables tiling.
- **vae_tile_size** / **vae_tile_overlap** — VAE pixel tile size and overlap,
  used only when `vae_tiling` decides to tile. `0` means `1024`.
- **noise_mode** — `reference` (default) takes the latent noise from the global
  CUDA generator after `manual_seed(seed)`, exactly as
  `inference_vosr_onestep.py` does, so reference results are reproducible. The
  global RNG state is saved and restored around the draw, so no other node in the
  workflow is affected. `isolated` uses a private CPU generator instead.
- **vae_tiling** — `auto` (default) runs the VAE single-pass — the reference
  behaviour, with no blending between tiles — up to 2048 px / 4.2 MP, and tiles
  above only to avoid OOM. `full` always runs it single-pass. `tiled` always
  tiles at `vae_tile_size`.

## Tiling is not optional above 512 px

VOSR 2.0 was trained natively at up to 512 px. **Whenever the upscaled output
exceeds 512×512, set `tile_size`** (e.g. `512`) — otherwise quality visibly
degrades. This isn't a performance knob at that resolution, it's required for
correct output.

The VAE is a separate question. It is purely convolutional (the Qwen 2D VAE has
no attention blocks at all) and the reference decodes the whole image in one
pass, so leave `vae_tiling` on `auto`. Tiling the VAE blends overlapping tiles
with a Gaussian mask, which is an approximation the reference never pays for;
`auto` only reaches for it above 2048 px, where the single pass stops fitting.

## Notes

- The node logs the resolved policy for each run (target size, noise mode, DiT
  tile, VAE single-pass vs tiled, alignment), so a completed run says which code
  path produced it.
- The node warns when the target resolution exceeds 512 px with DiT tiling
  disabled, but does not force tiling on automatically.
- `tile_overlap` must be smaller than `tile_size` (and likewise for the VAE
  pair); the node validates this before running.
- The pre-scale is Pillow's `BICUBIC`, matching the reference exactly, and the
  `image` input is therefore treated as what the reference treats it as: an
  8-bit image. A float input is rounded to 8 bits, exactly as saving it to PNG
  and running the reference would round it.
