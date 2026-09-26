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
#
# ---------------------------------------------------------------------------------------
# 实验分支 exp/sage-attention：给两个 fp16 站点接 SageAttention
#
# 精确数学路径在这块卡上很贵：4096 token、24 头的注意力，MATH 后端要物化 (24, 4096, 4096)
# 的分数矩阵，实测单次 482.9 ms；同样形状 SageAttention 只要 2.4 ms。DINOv2 那边是
# 14.0 ms -> 1.9 ms。这个文件是 VOSR2 所有注意力的唯一出口（dinov2.py、lightningdit.py、
# qwenimage_vae2d.py 都从这里 import），所以接线只需要动这一个文件。
#
# 默认关闭，行为与 main 完全一致；用 VOSR2_SAGE_ATTENTION=1 打开。
#
# VAE mid block 接不了 Sage：它是 (1, 1, 22500, 512) 的 fp32 单头注意力，Sage 的 kernel
# 要 131072 字节共享内存，硬件上限 65536，直接 OutOfResources。单头 512 通道也没法拆头
# 绕开（拆头会改变语义），所以它继续走 bounded_attention 的分块精确路径。
#
# Sage 是 8-bit 量化的注意力，不是零损失，所以开之前必须用真实输入对过最终成图。
# ---------------------------------------------------------------------------------------
import atexit
import os
import time

import torch
import torch.nn.functional as F

_OFF_VALUES = ("", "0", "false", "off", "no")
# 每次调用都读一遍环境变量：这样同一个进程里就能开关对比（A/B 跑两遍不用重启），
# 每次只多一次 os.environ.get，相对单次注意力几百毫秒可以忽略。
_disabled_after_error = False


def sage_wanted() -> bool:
    return not _disabled_after_error and \
        os.environ.get("VOSR2_SAGE_ATTENTION", "").strip().lower() not in _OFF_VALUES


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

    _sage_fn = None
    _sage_tried = False

    def _sageattn():
        """惰性加载 SageAttention；失败就永久回落，不反复重试、不刷日志。"""
        global _sage_fn, _sage_tried
        if _sage_tried:
            return _sage_fn
        _sage_tried = True
        try:
            from sageattention import sageattn

            _sage_fn = sageattn
            print("[vosr2] VOSR2_SAGE_ATTENTION=1：DINOv2 / LightningDiT 的注意力改走 SageAttention，"
                  "VAE mid block 仍是分块精确路径。")
        except Exception as exc:  # 没装 / 装坏 / 这个 arch 上的内核编译失败
            print(f"[vosr2] 想用 SageAttention 但加载失败（{type(exc).__name__}: {exc}），"
                  "两个 fp16 站点回落到精确数学路径。")
        return _sage_fn

    def _exact(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v)

    def scaled_dot_product_attention(q, k, v):
        global _disabled_after_error
        if sage_wanted() and q.dtype in (torch.float16, torch.bfloat16):
            fn = _sageattn()
            if fn is not None:
                try:
                    if PROFILE:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                    # ⚠️ 布局：这台机器上装的 sageattn 按 NHD 取数，即 (B, N, H, D)，
                    # 而 VOSR2 这三处传进来的都是 (B, H, N, D)。直接喂 HND 不会报错，
                    # 但输出跟正确结果完全无关（实测余弦相似度 0.008），最终图会变成方块乱码。
                    # 转成 NHD 再喂、结果转回 HND 之后，余弦 0.99992、最大差 0.018 —— 那点差
                    # 才是 int8 量化的真实代价。
                    # 注意这跟 ComfyUI 自带的 attention_sage（用 HND、传 tensor_layout=）
                    # 不是同一个 API 世代，别照抄那边的调用方式。
                    out = fn(q.transpose(1, 2).contiguous(),
                             k.transpose(1, 2).contiguous(),
                             v.transpose(1, 2).contiguous())
                    out = out.transpose(1, 2)
                    if PROFILE:
                        torch.cuda.synchronize()
                        _tick("sageattention", time.perf_counter() - t0)
                    return out
                except Exception as exc:
                    _disabled_after_error = True
                    print(f"[vosr2] sageattn 调用失败（{type(exc).__name__}: {str(exc)[:140]}），"
                          "本次及后续一律回落到精确数学路径。")
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

        Sage 在这条路上用不了（见文件头），所以这里永远走精确数学。
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
    def scaled_dot_product_attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def bounded_attention(q, k, v, max_score_elements=1 << 27):
        return F.scaled_dot_product_attention(q, k, v)
