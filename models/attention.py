# Attention dispatch for VOSR2's three attention sites: the two bf16 modules (DINOv2 and
# LightningDiT) and the fp32 VAE's mid block.
#
# PyTorch's ROCm attention kernels (AOTriton) are shipped per gfx arch, but torch still
# reports the flash / mem-efficient backends as available on arches it has no kernel image
# for. ComfyUI's `main.py` sets TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 on every start,
# which is what makes them selectable, and VOSR2 calls SDPA directly rather than going
# through comfy's attention selection.
#
# On gfx1103 they fail in two different ways. The LightningDiT/DINOv2 shapes die
# asynchronously instead of raising at the call, so the error surfaces at a later, unrelated
# op -- the `proj` Linear right after DINOv2's first attention, reported as
# "CUDA error: invalid argument" (hipErrorInvalidValue), which makes the culprit look like a
# broken matmul. The VAE mid block's 4-D (b, 1, N, c) fp32 call -- which upstream's comment
# says is *chosen* to select a fused kernel -- does not fail at all: it silently covers the
# whole image in a regular grid pattern.
#
# That grid is what a user sees at 1200px. It was traced here, not guessed: with the env var
# set, a bare script reproduces it; with the env var unset (same code, same input, same seed,
# same noise) the same script is clean and matches the fp32 reference to 1/255. The only
# SDPA call site the DiT-oriented pin did not cover was the VAE mid block.
#
# So nothing in VOSR2 uses a fused SDPA kernel on ROCm. `scaled_dot_product_attention` pins
# the exact-math backend for the small attentions (DINOv2's 1024 tokens, the DiT's 4096-token
# latent tiles). `bounded_attention` computes the same math chunked over the query axis for
# the VAE mid block, whose token count makes the O(N^2) score matrix the binding constraint
# (22500 tokens at 1200px, 262144 at 4096px -- a 2 GiB and a 274 GiB single-shot allocation).
import torch
import torch.nn.functional as F

if torch.version.hip is not None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def scaled_dot_product_attention(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v)

    def bounded_attention(q, k, v, max_score_elements=1 << 27):
        """Exact attention with a bounded score-matrix allocation.

        Same math as the pinned path -- fp32 scores, softmax over the whole key axis -- but
        computed one query block at a time, so peak allocation is `max_score_elements`
        instead of chunk x N. A single pinned call is used when the matrix already fits.
        """
        n = q.shape[-2]
        chunk = max(1, min(n, max_score_elements // max(n, 1)))
        if chunk >= n:
            return scaled_dot_product_attention(q, k, v)
        scale = q.shape[-1] ** -0.5
        blocks = []
        for start in range(0, n, chunk):
            scores = torch.matmul(q[..., start:start + chunk, :], k.transpose(-1, -2)) * scale
            blocks.append(torch.matmul(scores.softmax(dim=-1), v))
        return torch.cat(blocks, dim=-2)
else:
    def scaled_dot_product_attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def bounded_attention(q, k, v, max_score_elements=1 << 27):
        return F.scaled_dot_product_attention(q, k, v)
