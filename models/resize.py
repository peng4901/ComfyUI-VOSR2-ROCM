"""Reference pre-scale: Pillow's BICUBIC resize of the 8-bit input image.

Upstream's inference contract (``inference_vosr_onestep.py``) is

    input_img = raw_img.resize((w * upscale, h * upscale), Image.BICUBIC)
    lq = to_tensor(input_img).unsqueeze(0) * 2.0 - 1.0

so the LQ tensor VOSR 2.0 is defined against -- and was trained and demoed with -- is a
*bicubic upscale of an 8-bit PIL image*.

``torch.nn.functional.interpolate(mode="bicubic")`` is not a drop-in substitute for it:

* Pillow's bicubic kernel uses ``a = -0.5``; PyTorch's uses ``a = -0.75``.
* Pillow resizes 8-bit images through an 8-bit intermediate -- its horizontal and vertical
  passes each round back to uint8 -- while ``F.interpolate`` keeps one float pass.

Measured on this box (64x64 -> 2x, the reference's own input), the two differ by up to
7.2/255 per channel with only 51% of pixels equal. That is small in the LQ but the
one-step DiT amplifies it by roughly an order of magnitude, so it dominated the fork's
final-pixel deviation from the reference.

Reproducing Pillow's integer coefficient tables and its 8-bit intermediate would be a
faithful but fragile reimplementation, so this calls Pillow: the same implementation the
reference calls, on the same 8-bit data. It also means a ComfyUI IMAGE is treated as what
the reference treats it as -- an 8-bit image -- so a float input is rounded to 8 bits
exactly as saving it to PNG and running the reference would round it.

Verified: on the reference's own input this reproduces the recorded reference ``lq``
tensor bit-for-bit.
"""
import numpy as np
import torch
from PIL import Image

_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def prescale_bicubic(images_bhwc01: torch.Tensor, upscale: int, device) -> torch.Tensor:
    """Pillow BICUBIC ``upscale``x of a BHWC [0, 1] image -> BCHW [0, 1] float32 on `device`."""
    if upscale < 1:
        raise ValueError(f"VOSR2Upscale: upscale must be >= 1, got {upscale}.")
    if images_bhwc01.shape[-1] != 3:
        raise ValueError(f"VOSR2Upscale expects a 3-channel RGB IMAGE, got {images_bhwc01.shape[-1]} channels.")

    # Round-trip through uint8: the reference's LQ is an 8-bit image, and `k / 255` recovers
    # `k` exactly, so an 8-bit IMAGE (LoadImage's output) survives this unchanged.
    x = images_bhwc01.detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    u8 = (x.to(torch.float64) * 255.0).round().to(torch.uint8).numpy()

    b, h, w, _ = u8.shape
    resized = np.empty((b, h * upscale, w * upscale, 3), dtype=np.uint8)
    for i in range(b):
        resized[i] = np.asarray(Image.fromarray(u8[i]).resize((w * upscale, h * upscale), _BICUBIC))

    return torch.from_numpy(resized).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
