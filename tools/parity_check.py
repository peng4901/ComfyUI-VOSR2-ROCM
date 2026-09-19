"""Numeric parity against a captured upstream reference run.

VOSR 2.0 is a *one-step* model: a single DiT call turns `(lq_latent, noise)` into the SR
latent, so anything that perturbs the pre-scale, the noise, or the arithmetic is amplified
roughly tenfold at the output. "The image looks fine" is therefore not evidence of parity,
and neither is a single-number diff on the final PNG -- a stage-by-stage comparison is.

This runs the fork's own `run_vosr2` on the reference's own input and compares every
intermediate against a dump captured from the stock upstream script
(`_vosr_diag/a_dump.py`, itself validated against the untouched stock CLI to within this
box's 1/255 run-to-run noise), once per dtype.

    python tools/parity_check.py --comfyui <ComfyUI dir> --reference-dump <intermediates.pt> \
        --input <the reference's input PNG> --upscale 2 --dtypes fp32,bf16,fp16

Needs the ComfyUI environment (it imports `comfy.model_management` and `einops`); run it
with ComfyUI's own interpreter, not a system Python. It does not need ComfyUI to be
running. Exit status is 0 when the final image stays inside `--tolerance` default 2/255
for every dtype.
"""
import argparse
import importlib.util
import os
import sys
import tempfile


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", default=os.environ.get("COMFYUI_DIR", r"E:\ComfyUI_JZ\ComfyUI"),
                   help="ComfyUI checkout to import comfy.* from")
    p.add_argument("--pack", default=here, help="this custom-node package (default: the fork itself)")
    p.add_argument("--reference-dump", required=True)
    p.add_argument("--input", required=True, help="the PNG the reference dump was produced from")
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=0)
    p.add_argument("--tile-overlap", type=int, default=32)
    p.add_argument("--dtypes", default="fp32")
    p.add_argument("--tolerance", type=float, default=2.0, help="allowed final max |diff| in 0..255")
    p.add_argument("--save-png", default="", help="write each dtype's output here as <dtype>.png (for matched visuals)")
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def check_node_contract(nodes_module) -> str:
    """Every declared input must be an `execute` parameter, in the same order.

    ComfyUI dispatches by name, but the *frontend* maps a saved workflow's positional
    `widgets_values` onto the inputs in schema order, and `validate_inputs` is called with
    that order too. A schema/execute disagreement therefore stays silent until a workflow
    produces wrong output, which is exactly the failure mode adding inputs risks.
    """
    import inspect

    declared = [i.id for i in nodes_module.VOSR2Upscale.define_schema().inputs]
    params = [p for p in inspect.signature(nodes_module.VOSR2Upscale.execute).parameters if p != "cls"]
    if declared != params:
        return f"VOSR2Upscale schema {declared} != execute {params}"
    loader_inputs = [i.id for i in nodes_module.VOSR2ModelLoader.define_schema().inputs]
    loader_params = [p for p in inspect.signature(nodes_module.VOSR2ModelLoader.execute).parameters if p != "cls"]
    if loader_inputs != loader_params:
        return f"VOSR2ModelLoader schema {loader_inputs} != execute {loader_params}"
    return ""


def main() -> int:
    args = parse_args()

    # A ComfyUI install that has never run here may not have a MIOpen kernel cache where it
    # wants one; keep it inside a scratch dir rather than failing on a permissions error.
    work = args.out_dir or os.path.join(tempfile.gettempdir(), "vosr2-parity")
    os.makedirs(work, exist_ok=True)
    os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.join(work, "miopen"))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.environ["MIOPEN_USER_DB_PATH"])

    sys.path.insert(0, args.comfyui)
    sys.argv = ["main.py", "--port", "8189"]
    import comfy.options  # noqa: E402

    comfy.options.enable_args_parsing()

    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from PIL import Image  # noqa: E402

    spec = importlib.util.spec_from_file_location(
        "vosr2_pack", os.path.join(args.pack, "__init__.py"),
        submodule_search_locations=[args.pack],
    )
    pack = importlib.util.module_from_spec(spec)
    sys.modules["vosr2_pack"] = pack
    spec.loader.exec_module(pack)
    loader = sys.modules["vosr2_pack.loader"]
    inference = sys.modules["vosr2_pack.inference"]
    nodes = sys.modules["vosr2_pack.nodes_vosr"]

    contract = check_node_contract(nodes)
    if contract:
        print(f"[parity] FAILED node contract: {contract}")
        return 1
    print("[parity] node contract OK (schema order == execute order)")

    ref = torch.load(args.reference_dump, map_location="cpu", weights_only=False)
    ref_lq = ref["lq"].to(torch.float32).cpu()
    ref_pixels = ref["pixels"].permute(1, 2, 0).numpy().astype(np.int16)

    img = np.asarray(Image.open(args.input).convert("RGB")).astype(np.float32) / 255.0
    image = torch.from_numpy(img)[None]

    print(f"[parity] reference dump {os.path.abspath(args.reference_dump)}")
    print(f"[parity] reference meta upscale={ref['meta']['upscale']} seed={ref['meta']['seed']} "
          f"align={ref['meta']['align_method']} lq={tuple(ref_lq.shape)}")
    print(f"[parity] pack {os.path.abspath(args.pack)}")
    print()

    stages = [
        ("lq (pre-scale)", lambda c: c["resized01"] * 2.0 - 1.0, ref_lq),
        ("lq_latent", lambda c: c["lq_latent"], ref["lq_latent"]),
        ("noise", lambda c: c["noise_in"], ref["noise"]),
        ("DINOv2 layer17", lambda c: c["venc"][0], ref["venc_layer17"]),
        ("DiT velocity u", lambda c: c["u"], ref["calls"][0]["u"]),
        ("sr_latent", lambda c: c["sr_latent"], ref["sr_latent"]),
        ("sr_tensor", lambda c: c["sr_tensor"], ref["sr_tensor"]),
    ]

    failures = []
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    for dtype in dtypes:
        print(f"=== dtype={dtype} ===")
        real_resolve = loader._resolve_dtype
        forced = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
        loader._resolve_dtype = (lambda d, dev, _f=forced: _f) if forced else real_resolve

        model = loader.load_vosr2("VOSR2", dtype)
        VOSR2Model = type(model)
        cap = {}

        original = {name: getattr(VOSR2Model, name)
                    for name in ("vision_features", "dit_velocity", "denoise_one_step")}

        def vision_features(self, lq, _o=original["vision_features"]):
            out = _o(self, lq)
            cap["venc"] = [t.detach().to("cpu", torch.float32) for t in out]
            return out

        def dit_velocity(self, inp, t_cur, t_next, venc_fea, _o=original["dit_velocity"]):
            out = _o(self, inp, t_cur, t_next, venc_fea)
            cap["u"] = out.detach().to("cpu", torch.float32)
            return out

        def denoise_one_step(self, lq_latent, noise, venc_fea, _o=original["denoise_one_step"]):
            cap["lq_latent"] = lq_latent.detach().to("cpu", torch.float32)
            cap["noise_in"] = noise.detach().to("cpu", torch.float32)
            out = _o(self, lq_latent, noise, venc_fea)
            cap["sr_latent"] = out.detach().to("cpu", torch.float32)
            return out

        VOSR2Model.vision_features = vision_features
        VOSR2Model.dit_velocity = dit_velocity
        VOSR2Model.denoise_one_step = denoise_one_step
        try:
            out = inference.run_vosr2(
                model, image, args.upscale, args.seed, "none",
                args.tile_size, args.tile_overlap, 0, 32, "reference", "full",
            )
            torch.cuda.synchronize()
        finally:
            for name, fn in original.items():
                setattr(VOSR2Model, name, fn)
            loader._resolve_dtype = real_resolve

        cap["resized01"] = inference._resize_to_target(image, args.upscale,
                                                      torch.device("cuda")).detach().to(torch.float32)
        cap["sr_tensor"] = out.movedim(-1, 1) * 2.0 - 1.0
        px = (out.clamp(0, 1)[0].cpu() * 255.0 + 0.5).to(torch.uint8).permute(2, 0, 1)

        print(f"{'stage':<18}{'max':>12}{'mean':>12}")
        for name, get, target in stages:
            got = get(cap)
            expected = target.to(torch.float32).cpu()
            if tuple(got.shape) != tuple(expected.shape):
                print(f"{name:<18}{'shape mismatch':>12}  {tuple(got.shape)} vs {tuple(expected.shape)}")
                continue
            d = (got.to(torch.float32).cpu() - expected).abs()
            print(f"{name:<18}{d.max().item():>12.5f}{d.mean().item():>12.5f}")

        d8 = np.abs(px.permute(1, 2, 0).numpy().astype(np.int16) - ref_pixels)
        print(f"{'final pixels/255':<18}{int(d8.max()):>12}{float(d8.mean()):>12.3f}")
        ok = d8.max() <= args.tolerance
        print(f"-> {'PASS' if ok else 'FAIL'} (tolerance {args.tolerance}/255)\n")
        if not ok:
            failures.append((dtype, int(d8.max()), float(d8.mean())))

        if args.save_png:
            os.makedirs(args.save_png, exist_ok=True)
            Image.fromarray(px.permute(1, 2, 0).numpy()).save(
                os.path.join(args.save_png, f"fork_{dtype}.png"))
            if dtype == dtypes[0]:
                Image.fromarray(ref_pixels.astype(np.uint8)).save(
                    os.path.join(args.save_png, "reference.png"))
                Image.fromarray(np.asarray(Image.open(args.input).convert("RGB"))).save(
                    os.path.join(args.save_png, "source.png"))

        if dtype == dtypes[0]:
            # The node wrapper is a separate surface from run_vosr2: declared inputs,
            # validate_inputs and execute() all have to agree, and a mismatch there is
            # silent -- ComfyUI would simply pass the wrong argument.
            nodes = sys.modules["vosr2_pack.nodes_vosr"]
            wrapper = nodes.VOSR2Upscale
            invalid = wrapper.validate_inputs(
                model, image, args.upscale, args.seed, "none",
                args.tile_size, args.tile_overlap, 0, 32, "reference", "full",
            )
            node_out = wrapper.execute(
                model, image, args.upscale, args.seed, "none",
                args.tile_size, args.tile_overlap, 0, 32, "reference", "full",
            ).result[0]
            dn = (node_out.movedim(-1, 1).to(torch.float32).cpu()
                  - out.movedim(-1, 1).to(torch.float32).cpu()).abs().max().item() * 255.0
            print(f"node wrapper: validate_inputs={invalid!r}, "
                  f"VOSR2Upscale.execute vs run_vosr2 = {dn:.3f}/255")
            if invalid is not True or dn > 1.0:
                failures.append((f"{dtype}/node-wrapper", round(dn), 0.0))
            print()

        del model, cap
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if failures:
        print("[parity] FAILED:", ", ".join(f"{d}: max {m} mean {mn:.3f}" for d, m, mn in failures))
        return 1
    print("[parity] all dtypes within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
