[English](README.md) | **简体中文**

# ComfyUI-VOSR2-ROCM

[ComfyUI-VOSR2](https://github.com/ylchen333/ComfyUI-VOSR2) 的 AMD ROCm 分支，用于 **VOSR 2.0**
—— 单步（one-step）、1.4B 的图像超分（LightningDiT + Qwen-Image 2D VAE + DINOv2-L）。

上游只在 CPU 与 CUDA 上开发和做 CI。在 ROCm 上这个节点开箱即用是跑不通的：它会在视觉编码器里
崩溃；即使跑起来，放大后的画面上也会出现规则的网格状伪影。本分支把这两个问题都修掉了，并且能在
本机 1/255 的噪声底之内复现上游参考实现的结果。

- 上游节点：<https://github.com/ylchen333/ComfyUI-VOSR2>
- 官方 VOSR 代码：<https://github.com/cswry/VOSR>
- 官方权重：<https://huggingface.co/CSWRY/VOSR>

## 本分支改了什么

### 1. 注意力后端

ComfyUI 的 `main.py` 每次启动都会设置 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`，正是这个
环境变量让 PyTorch 的融合 SDPA 后端变得"可选"。在 `gfx1103` 上 AOTriton 的内核镜像缺失或不
适用，并且以两种不同的方式失效：

| 调用位置 | 失效方式 |
|---|---|
| LightningDiT 与 DINOv2 | 能启动，但要等到**下一次** CUDA 调用才异步报错 —— 表现为 DINOv2 `proj` Linear 处的 `CUDA error: invalid argument` |
| VAE mid block（4 维 `(b, 1, N, c)`，fp32） | **不报错**。返回的整幅图覆盖一层规则网格 —— 512 px 看不出来，1200 px 非常明显 |

`models/attention.py` 是唯一决定后端策略的地方。**两个 fp16 站点会优先启用 SageAttention**，不
可用时回落到精确数学路径；VAE mid block 用同样的精确数学按 query 分块计算。非 ROCm 设备仍然走
PyTorch 的正常分发，CUDA 输出不变。

注意：**融合 SDPA 后端在 ROCm 上不会被尝试，连回退都不算** —— 它们在这块卡上不保证抛异常
（网格伪影就是融合内核静默返回错误数值），而异步失败会甩到后面某个无关的算子上，`try/except`
兜不住。SageAttention 用不了时就直接走精确路径。VAE mid block 也接不了 Sage：它是
`(1, 1, 22500, 512)` 的 fp32 单头注意力，内核要 131072 字节共享内存，硬件上限 65536。

本机实测（780M / gfx1103，150px → 4×，`tile_size=512`，fp16，同种子）：精确路径 12.6 s，
SageAttention **9.3 s**，相对精确路径最差像素 82/255、平均差 0.78/255。大约 1.2~1.35×，
代价落在跟 bf16 同一量级，是取舍不是白捡。`VOSR2_ATTENTION=exact` 可以关掉。

### 2. 参考可复现性

VOSR 2.0 是单步模型：低质输入、潜空间噪声或运算上的微小差异，都会在成图上被放大。ComfyUI 移植
版原先有三处与上游推理脚本不同，现在都改成了显式选项：

| 契约 | 上游 | 本分支 |
|---|---|---|
| 预缩放 | 对 8 bit 图像做 `Image.BICUBIC` | 相同 —— `models/resize.py` 在同一份 8 bit 数据上调用 Pillow |
| 潜空间噪声 | `manual_seed(seed)` 之后从全局 CUDA RNG 做 `torch.randn_like(latent)` | `noise_mode=reference`（默认）：同一抽样，抽完立即还原全局 CPU/CUDA 状态，不影响其他节点 |
| VAE 分块 | 除非设置 `--vae_tile_size`，否则单次整图 | `vae_tiling=auto`（默认）：2048 px / 4.2 MP 以内单次整图，超过才分块以避免 OOM |

在 fp32 加上述策略下，参考的预缩放输入、潜变量与噪声逐位一致，最终成图差异在 **1/255** 以内
—— 与在本机上把参考实现跑两遍之间的差异相同。

### 3. 精度

上游实现全程 fp32。本分支提供 `fp32` 选项，并实测了低精度路径相对 fp32 参考的偏差
（128 px → 4×，输出 512²）：

| `dtype` | 相对 fp32 参考的成图差异 |
|---|---|
| `fp32` | 最大 1/255，平均 0.50/255 |
| `fp16` | 最差像素 32/255，平均 0.59/255 |
| `bf16` | 最差像素 94/255，平均最高 5.73/255 —— 不建议用于该模型 |

`default` 跟随 ComfyUI 的设备策略（`unet_dtype()`），在本机上即 fp16。需要复现参考结果时请用
`fp32`。

## 环境要求

- ComfyUI + AMD GPU + ROCm 版 PyTorch。
- `einops`、`safetensors`、`huggingface_hub`，ComfyUI 均已自带。
- 实测环境：AMD Radeon 780M（`gfx1103`）、torch 2.11.0+rocm10.0.0、ComfyUI 0.36。
- 模型文件共约 7 GB，首次运行时下载（见下）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peng4901/ComfyUI-VOSR2-ROCM
# 重启 ComfyUI
```

首次运行会从 [`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR) 把 VOSR 2.0 bundle 下载到
`ComfyUI/models/vosr2/`。手工维护安装时，建议把 `custom_nodes/ComfyUI-VOSR2-ROCM` 做成指向你
工作目录的符号链接或 junction：这样改代码只需重启 ComfyUI，不必重新安装。

## 节点

### VOSR 2.0 模型加载器（`VOSR2ModelLoader`）

| 输入 | 默认 | 说明 |
|---|---|---|
| `model` | `VOSR2` | `models/vosr2/` 下的 bundle 目录 —— 包含 DiT 及其配套的 VAE 与视觉编码器 |
| `dtype` | `default` | DiT 与视觉编码器的精度：`default` / `fp16` / `bf16` / `fp32`。VAE 始终以 fp32 运行 |

VOSR 2.0 是一个固定三元组：DiT 只能搭配它训练时使用的那个 Qwen 2D VAE，条件编码也指定到
DINOv2-L 的某一层。因此 VAE 与视觉编码器是 bundle 的内部组成，而不是独立输入。

### VOSR 2.0 放大（`VOSR2Upscale`）

| 输入 | 默认 | 范围 | 说明 |
|---|---:|---|---|
| `model` | — | `VOSR2_MODEL` | 来自上面的加载器 |
| `image` | — | `IMAGE` | 单图或批次 |
| `upscale` | `4` | ≥ 1，不限上限 | 精确输出倍数；约 4× 是文档给出的最佳区间 |
| `seed` | `42` | ≥ 0 | 潜空间噪声种子（行为见 `noise_mode`） |
| `color_alignment` | `wavelet` | `wavelet` / `adain` / `none` | 相对双三次目标做后处理对齐 |
| `tile_size` | `0` | `0`–`4096`，步进 64 | DiT 像素分块；`0` 表示不分块 |
| `tile_overlap` | `32` | `0`–`512`，步进 8 | DiT 分块重叠 |
| `vae_tile_size` | `0` | `0`–`8192`，步进 64 | `vae_tiling` 决定分块时使用的 VAE 像素块；`0` 表示 1024 |
| `vae_tile_overlap` | `32` | `0`–`512`，步进 8 | VAE 分块重叠 |
| `noise_mode` | `reference` | `reference` / `isolated` | `reference` 复现上游的抽样方式；`isolated` 用私有生成器且不改动全局 RNG 状态（批次第 *i* 张用 `seed + i`） |
| `vae_tiling` | `auto` | `auto` / `full` / `tiled` | `auto` 在 2048 px / 4.2 MP 以内对 VAE 单次整图（即参考行为，无分块融合误差）；`tiled` 始终按 `vae_tile_size` 分块 |

**超过 512 px 时，分块不是可选项。** VOSR 2.0 的训练分辨率上限是 512 px，所以只要*放大后*的尺寸
超过 512×512，就要设置 `tile_size`（例如 `512`），否则质量会明显下降。

## 模型文件

bundle 位于 `ComfyUI/models/vosr2/<bundle>/`：

```text
models/vosr2/VOSR2/
    args.json
    checkpoints/ema_model.safetensors
    Qwen-Image-vae-2d/{config.json, diffusion_pytorch_model.safetensors}
    dinov2_vitl14.safetensors
```

加载器会在首次使用时从 `CSWRY/VOSR` 补齐缺失文件，并在构建任何模型之前把 `args.json` 与固定的
VOSR 2.0 架构做校验。离线安装可按下表手工放置：

| `CSWRY/VOSR` 中的文件 | 放置到 |
|---|---|
| [`VOSR2/args.json`](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/args.json) | `models/vosr2/VOSR2/args.json` |
| [`VOSR2/checkpoints/ema_model.safetensors`](https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR2/checkpoints/ema_model.safetensors) | `models/vosr2/VOSR2/checkpoints/ema_model.safetensors` |
| [`Qwen-Image-vae-2d/config.json`](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/config.json) | `models/vosr2/VOSR2/Qwen-Image-vae-2d/config.json` |
| [`Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors`](https://huggingface.co/CSWRY/VOSR/resolve/main/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors) | `models/vosr2/VOSR2/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors` |
| [`torch_cache/checkpoints/dinov2_vitl14_pretrain.pth`](https://huggingface.co/CSWRY/VOSR/resolve/main/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth) | 转换成 `models/vosr2/VOSR2/dinov2_vitl14.safetensors`（自动下载时加载器会自行完成转换） |

## 显存与分块

- `tile_size` 约束 DiT 的峰值显存，`vae_tiling` 约束 VAE 的峰值显存，两者相互独立。
- 在 ROCm 上所有注意力都走精确数学路径，其打分矩阵随 token 数**平方**增长。这就是本平台上
  `tile_size` 比在 CUDA 上更重要的原因：1200 px 不分块大约需要 48 GiB 的注意力中间量，所以 DiT
  分块在这里是必需项而不是优化项。
- VAE 是纯卷积结构，`vae_tiling=auto` 在常规尺寸下保持单次整图，比融合分块既更快也更接近参考。
  详见 [`docs/avoiding-oom.md`](docs/avoiding-oom.md)。

## 验证

上面所有结论都是实测的，不是推断。参考基准是同机 fp32 运行的原始上游
`inference_vosr_onestep.py`，并捕获了每个中间量；本分支逐阶段对齐复现：

| 阶段 | fp32、策略一致时 |
|---|---|
| 预缩放输入、潜变量、潜空间噪声 | 逐位一致 |
| DINOv2 第 17 层条件特征 | 最大 1.2e-4 |
| DiT 速度场、SR 潜变量 | 最大 1.3e-4 |
| 最终成图，不分块 512² 与 1024² | 最大 1/255，平均 0.50/255 |
| 最终成图，分块 8×（3×3 融合网格） | 最大 1/255，平均 0.50/255 |

注意力修复背后的隔离实验见 [`docs/ROCm.md`](docs/ROCm.md)。

## 许可

节点代码为 Apache-2.0（见 [`LICENSE`](LICENSE)）。模型权重由
[`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR) 按其自身条款分发；其中 DINOv2-L 权重为
CC-BY-NC 4.0（禁止商用）。任何商业用途前请先阅读这些条款。

## 致谢

- **VOSR / VOSR 2.0** — Rongyuan Wu 等（[cswry/VOSR](https://github.com/cswry/VOSR)）
- **ComfyUI 节点** — [ylchen333/ComfyUI-VOSR2](https://github.com/ylchen333/ComfyUI-VOSR2)
- **DINOv2** — Meta AI（[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)）
