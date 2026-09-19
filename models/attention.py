# Attention dispatch for the two bf16 modules (DINOv2 and LightningDiT).
#
# PyTorch's ROCm attention kernels (AOTriton) are shipped per gfx arch, but torch
# still reports the flash / mem-efficient backends as available on arches it has
# no kernel image for. The launch then fails asynchronously instead of raising at
# the call, so the error surfaces at a later, unrelated op -- on gfx1103 that is
# the `proj` Linear right after the first attention, reported as
# "CUDA error: invalid argument" (hipErrorInvalidValue), which makes the real
# culprit look like a broken matmul in DINOv2.
#
# ComfyUI itself launches with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, so those
# kernels are live on every ROCm box, and VOSR2 calls SDPA directly rather than
# going through comfy's attention selection. Use the exact-math backend on ROCm:
# the VOSR2 attentions are small (1024 tokens, 64-wide heads, MLP-dominated
# layers), so this costs little next to a kernel that silently poisons the
# context.
#
# The fp32 VAE is deliberately left on the default backends: its tiles reach
# ~16k tokens, where MATH's O(N^2) attention matrix would be a real memory cost,
# and fp32 does not select the broken kernels (verified on gfx1103).
import torch

if torch.version.hip is not None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def scaled_dot_product_attention(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
else:
    def scaled_dot_product_attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
