# Attention dispatch for VOSR2's three attention sites: the two fp16 modules (DINOv2 and
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
# ---------------------------------------------------------------------------------------
# Which kernel each site gets
#
#   CUDA  : torch's own SDPA dispatch, which already prefers flash -> mem-efficient -> math.
#           Left alone; there is nothing to improve and no silent-failure history here.
#   ROCm  : SageAttention when it is installed, usable, and passes a one-off self-test;
#           otherwise the exact-math backend. The fused SDPA backends are NOT tried on
#           ROCm even as a fallback -- see the note below.
#   VAE   : always the exact chunked path. Sage's kernel needs 131072 bytes of shared memory
#           for this site's (1, 1, 22500, 512) fp32 single-head shape and the hardware limit
#           is 65536, so it simply cannot take it.
#
# Why the fused backends are not attempted on ROCm: on this arch they do not reliably raise.
# The VAE grid artifact was a fused kernel returning wrong numbers with no error at all, and
# a failed async kernel launch surfaces at whatever unrelated op runs next -- so a
# try/except around a probe call cannot contain it. Preferring them "when available" would
# mean silently preferring a known-broken path. `VOSR2_ATTENTION=exact` forces the old
# behaviour everywhere.
#
# The self-test exists because a wrong-layout or wrong-API kernel can return plausible
# garbage without raising. Measured example: this build of sageattn takes NHD (B, N, H, D),
# not the HND (B, H, N, D) layout ComfyUI's own attention_sage uses. Feeding HND returned a
# result with cosine similarity 0.008 to the correct answer and produced a blocky mess --
# no exception, no warning. The self-test compares against the exact reference on the real
# shape and refuses anything under 0.99.
# ---------------------------------------------------------------------------------------
import atexit
import os
import time

import torch
import torch.nn.functional as F

_OFF_VALUES = ("", "0", "false", "off", "no")


def _exact_requested() -> bool:
    """VOSR2_ATTENTION=exact 强制走精确数学路径（老行为）。

    VOSR2_SAGE_ATTENTION=0 也认，那是试着接 Sage 时留下的开关。
    """
    for key in ("VOSR2_ATTENTION", "VOSR2_SAGE_ATTENTION"):
        v = os.environ.get(key)
        if v is None:
            continue
        v = v.strip().lower()
        if v in ("exact", "off", "0", "false", "no"):
            return True
    return False


PROFILE = os.environ.get("VOSR2_PROFILE", "").strip().lower() not in _OFF_VALUES
_STATS: dict = {}


def _tick(name: str, seconds: float) -> None:
    row = _STATS.setdefault(name, [0, 0.0])
    row[0] += 1
    row[1] += seconds


@atexit.register
def _dump_stats() -> None:
    """VOSR2_PROFILE=1 时，退出前把每个注意力站点的调用次数与累计耗时打出来。"""
    if not PROFILE or not _STATS:
        return
    print("[vosr2-profile] 注意力调用统计（含每次 cuda.synchronize，用于对比而非绝对性能）：")
    for name, (n, total) in sorted(_STATS.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:26} {n:5d} 次  {total * 1000:9.1f} ms  平均 {total / max(n, 1) * 1000:7.2f} ms")


if torch.version.hip is not None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _KERNEL = None          # None = 还没定 | "sage" | "exact"
    _KERNEL_NOTE = ""

    def _exact(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v)

    def _sage_call(q, k, v):
        """调用这套 build 的 sageattn。

        ⚠️ 布局：这台机器上的 sageattn 按 NHD 取数，即 (B, N, H, D)，而 VOSR2 传进来的是
        (B, H, N, D)。喂错不报错，但结果是垃圾（实测余弦 0.008，图变方块乱码）。
        转成 NHD 喂、结果转回 HND 之后余弦 0.99992。这跟 ComfyUI 自带的 attention_sage
        （用 HND、会传 tensor_layout=）不是同一个 API 世代，别照抄那边的写法。
        """
        from sageattention import sageattn

        return sageattn(q.transpose(1, 2).contiguous(),
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous()).transpose(1, 2)

    def _selftest(fn, q, k, v, min_cos: float = 0.99, max_rel: float = 0.05) -> bool:
        """拿真实形状和 dtype 跑一次，跟精确参考比。只跑一次，毫秒级。

        两个刻意的设计，都是踩过坑之后加的：

        1. **用独立的随机 q/k/v，不用传进来的那几个。** 一开始图省事复用了真实张量，而调用
           方经常是 ``attn(x)`` 里 q=k=v 同源，这时注意力权重被对角项主导，输出≈输入本身，
           于是一个布局完全喂反的内核也能拿到很高的余弦相似度 —— 自检形同虚设。独立随机
           输入才能让"结果是不是真算出来的"变得可判。
        2. **余弦之外还看相对幅度。** 余弦是尺度无关的，一个方向对、幅度差十倍的内核能轻松
           过线；这里再要求最大逐元素偏差不超过参考峰值的 5%。

        它专门拦"不报错但算错"的内核（布局喂反、API 对不上）。至于量化误差这类数据相关的
        质量差异，自检判不了，那是 A/B 对比要回答的问题。
        """
        n = min(q.shape[-2], 128)
        if n < 2:
            return False
        shape = (q.shape[0], q.shape[1], n, q.shape[-1])
        qs = torch.randn(shape, device=q.device, dtype=q.dtype)
        ks = torch.randn(shape, device=q.device, dtype=q.dtype)
        vs = torch.randn(shape, device=q.device, dtype=q.dtype)
        try:
            got = fn(qs, ks, vs)
        except Exception:
            return False
        with sdpa_kernel([SDPBackend.MATH]):
            want = F.scaled_dot_product_attention(qs.float(), ks.float(), vs.float())
        if got.shape != want.shape or not torch.isfinite(got).all():
            return False
        g, w = got.float().reshape(-1), want.reshape(-1)
        cos = F.cosine_similarity(g, w, dim=0).item()
        rel = (g - w).abs().max().item() / (w.abs().max().item() + 1e-6)
        return cos >= min_cos and rel <= max_rel

    def _select(q, k, v) -> str:
        """第一次调用时定下用哪个内核，之后不变。"""
        global _KERNEL, _KERNEL_NOTE
        if _KERNEL is not None:
            return _KERNEL
        if _exact_requested():
            _KERNEL, _KERNEL_NOTE = "exact", "VOSR2_ATTENTION=exact"
            return _KERNEL
        if q.dtype not in (torch.float16, torch.bfloat16):
            _KERNEL, _KERNEL_NOTE = "exact", f"dtype={q.dtype} 不是 fp16/bf16，Sage 需要半精度"
            print(f"[vosr2] {_KERNEL_NOTE}，走精确数学路径。")
            return _KERNEL
        try:
            import sageattention  # noqa: F401
        except Exception as exc:
            _KERNEL, _KERNEL_NOTE = "exact", f"sageattention 不可用（{type(exc).__name__}）"
            print(f"[vosr2] {_KERNEL_NOTE}，走精确数学路径。")
            return _KERNEL
        if _selftest(_sage_call, q, k, v):
            _KERNEL, _KERNEL_NOTE = "sage", "SageAttention 自检通过（与精确参考余弦 ≥ 0.99）"
            print("[vosr2] DINOv2 / LightningDiT 的注意力走 SageAttention；"
                  "VAE mid block 仍是分块精确路径。")
        else:
            _KERNEL, _KERNEL_NOTE = "exact", "SageAttention 自检没过，永久回落到精确数学路径"
            print(f"[vosr2] {_KERNEL_NOTE}（自检余弦 < 0.99）。")
        return _KERNEL

    def scaled_dot_product_attention(q, k, v):
        kernel = _select(q, k, v)
        if kernel == "sage":
            try:
                if PROFILE:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                out = _sage_call(q, k, v)
                if PROFILE:
                    torch.cuda.synchronize()
                    _tick("sageattention", time.perf_counter() - t0)
                return out
            except Exception as exc:
                # 自检过了但真跑挂了：整段退回去，别在同一张图里混用两种算法。
                global _KERNEL, _KERNEL_NOTE
                _KERNEL, _KERNEL_NOTE = "exact", f"sageattn 运行期失败（{type(exc).__name__}）"
                print(f"[vosr2] {_KERNEL_NOTE}，本次及后续回落到精确数学路径。")
        if PROFILE:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = _exact(q, k, v)
        if PROFILE:
            torch.cuda.synchronize()
            _tick("sdpa-math 精确", time.perf_counter() - t0)
        return out

    def bounded_attention(q, k, v, max_score_elements=1 << 27):
        """Exact attention with a bounded score-matrix allocation.

        Same math as the pinned path -- fp32 scores, softmax over the whole key axis -- but
        computed one query block at a time, so peak allocation is `max_score_elements`
        instead of chunk x N. A single pinned call is used when the matrix already fits.

        这条路上接不了 SageAttention（见文件头：fp32 单头 512 通道超出共享内存上限），
        所以 VAE 永远走精确数学。
        """
        n = q.shape[-2]
        chunk = max(1, min(n, max_score_elements // max(n, 1)))
        if chunk >= n:
            if PROFILE:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            out = _exact(q, k, v)
            if PROFILE:
                torch.cuda.synchronize()
                _tick("vae-mid 精确(单次)", time.perf_counter() - t0)
            return out
        scale = q.shape[-1] ** -0.5
        blocks = []
        if PROFILE:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        for start in range(0, n, chunk):
            scores = torch.matmul(q[..., start:start + chunk, :], k.transpose(-1, -2)) * scale
            blocks.append(torch.matmul(scores.softmax(dim=-1), v))
        out = torch.cat(blocks, dim=-2)
        if PROFILE:
            torch.cuda.synchronize()
            _tick(f"vae-mid 精确({(n + chunk - 1) // chunk} 块)", time.perf_counter() - t0)
        return out
else:
    # CUDA and friends: torch's own dispatch already prefers flash -> mem-efficient -> math,
    # so there is nothing to pin here.
    def scaled_dot_product_attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def bounded_attention(q, k, v, max_score_elements=1 << 27):
        return F.scaled_dot_product_attention(q, k, v)
