"""
Converts the VSR-Dataset-Creator model checkpoints to ONNX for the dataset
nodes, and checks each export against the PyTorch model on one image.

    python scripts/export_dataset_models.py --models-dir <VSR-Dataset-Creator models> \
        --out-dir <where the .onnx files go> [--only denoise anicc prism prism_small] [--image picture.png]

This is a one-off conversion run by hand. It needs torch, spandrel, safetensors,
einops, onnx and onnxruntime, none of which the app ships. The network
definitions (GaterV3 where spandrel lacks it, BGCC v4 and the Prism SCUNet)
are copies from VSR-Dataset-Creator in scripts/dataset_archs/. The .onnx
files are written outside the repository: the weights come with their own
terms (see include/vk_dataset/NOTICE.md).

The exported models take their inputs by the names the nodes use:
    denoise, prism, prism_small   "input" (1, C, H, W) -> "output"
    anicc                         "hr" (1, 3, H, W), "lr" (1, 3, h, w) -> "output"
All sizes are dynamic. Prism takes four channels, RGB plus a constant plane
holding the CRF strength / 51.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

MODELS = {
    "denoise": "1x_bndl_animedenoise_v3.safetensors",
    "anicc": "2x_bndl_anicc.safetensors",
    "prism": "prism_h264_scunet.safetensors",
    "prism_small": "prism_h264_scunet_small.safetensors",
}


def load_state_dict(path):
    import torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            return state[key]
    return state


def build_denoiser(sd):
    """Spandrel's model for the checkpoint, or the GaterV3 network when spandrel does not know it."""
    try:
        from spandrel import ModelLoader
        descriptor = ModelLoader().load_from_state_dict(sd)
        if descriptor.scale != 1:
            raise SystemExit(f"the denoise model must be 1x, but it is {descriptor.scale}x")
        return descriptor.model, f"spandrel {descriptor.architecture.name}"
    except Exception as spandrel_error:
        if "in_to_dim.weight" not in sd:
            raise SystemExit(f"spandrel could not load the denoise model: {spandrel_error}")
    from dataset_archs.gaterv3 import GateRV3

    def block_counts(prefix):
        found = defaultdict(set)
        pattern = re.compile(rf"{prefix}\.(\d+)\.gated\.(\d+)\.")
        for key in sd:
            match = pattern.match(key)
            if match:
                found[int(match.group(1))].add(int(match.group(2)))
        return [max(v) + 1 for _, v in sorted(found.items())]

    meta = sd.get("dim_to_in.MetaUpsample")
    scale, upsample, mid_dim = 1, "pixelshuffle", 48
    if meta is not None:
        values = [int(v) for v in meta.flatten().tolist()]
        upsample = ("conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample")[values[1]]
        scale, mid_dim = values[2], values[5]
    if scale != 1:
        raise SystemExit(f"the denoise model must be 1x, but it is {scale}x")
    model = GateRV3(
        in_ch=sd["in_to_dim.weight"].shape[1], dim=sd["in_to_dim.weight"].shape[0],
        enc_blocks=tuple(block_counts("gater_encode")), dec_blocks=tuple(block_counts("decode")),
        num_latent=len({k.split(".")[1] for k in sd if k.startswith("latent.")}),
        sisr_blocks=len({k.split(".")[1] for k in sd if k.startswith("span_n_b.")}),
        scale=scale, upsample=upsample, upsample_mid_dim=mid_dim,
        attention=any("token_mix.qkv" in k for k in sd))
    model.load_state_dict(sd)
    return model, "GateRV3"


def build_anicc(sd):
    from dataset_archs.bgcc_v4 import BGCCv4
    model = BGCCv4(
        feat=sd["enc0.0.conv.weight"].shape[0],
        n_blocks=len({k.split(".body.")[1].split(".")[0] for k in sd if k.startswith("enc0.1.body.")}),
        num_slots=sd["router.slots"].shape[1], slot_dim=sd["router.slots"].shape[2],
        n_knots=(sd["global_align.mlp.2.weight"].shape[0] - 12) // 3)
    model.load_state_dict(sd)
    return model, "BGCC v4"


def build_prism(sd):
    from dataset_archs.scunet import PrismSCUNet
    dim = int(sd["m_head.0.weight"].shape[0])
    heads = int(sd["m_down1.0.trans_block.msa.relative_position_params"].shape[0])
    config = []
    for stage in ("m_down1", "m_down2", "m_down3", "m_body", "m_up3", "m_up2", "m_up1"):
        config.append(len({int(key.split(".")[1]) for key in sd
                           if key.startswith(stage + ".") and ".trans_block." in key}))
    model = PrismSCUNet(in_nc=int(sd["m_head.0.weight"].shape[1]), out_nc=int(sd["m_tail.0.weight"].shape[0]),
                        dim=dim, head_dim=(dim // 2) // heads, config=tuple(config))
    model.load_state_dict(sd)
    return model, f"Prism SCUNet dim={dim}"


def test_image(path, size=256):
    """A float RGB array (H, W, 3) in 0-1: the picture's top-left crop, or smooth texture plus grain."""
    import cv2
    import numpy as np
    if path:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[:size, :size].astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 1, (size, size, 3)).astype(np.float32), (0, 0), 3.0)
    base = (base - base.min()) / (base.max() - base.min())
    return np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1).astype(np.float32)


def export(name, model, inputs, input_names, out_path):
    """Exports with dynamic height and width, then compares onnxruntime with PyTorch on `inputs`."""
    import numpy as np
    import onnxruntime
    import torch

    model.eval()
    axes = {n: {2: f"{n}_h", 3: f"{n}_w"} for n in input_names}
    axes["output"] = {2: "out_h", 3: "out_w"}
    with torch.inference_mode():
        expected = model(*inputs).numpy()
    torch.onnx.export(model, tuple(inputs), out_path, input_names=input_names, output_names=["output"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {n: t.numpy() for n, t in zip(input_names, inputs)})[0]
    diff = float(np.abs(got - expected).max())
    print(f"{name}: {os.path.getsize(out_path) / 1e6:.1f} MB, output {tuple(got.shape)}, "
          f"max abs diff torch vs onnx {diff:.3e} (outputs span {expected.min():.3f}..{expected.max():.3f})")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", required=True, help="the folder with its .safetensors files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--image", help="a picture to compare on (default: synthetic grain)")
    args = parser.parse_args()

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.out_dir, exist_ok=True)
    image = torch.from_numpy(test_image(args.image).transpose(2, 0, 1)[None].copy())
    for name in args.only:
        sd = load_state_dict(os.path.join(args.models_dir, MODELS[name]))
        out = os.path.join(args.out_dir, MODELS[name].rsplit(".", 1)[0] + ".onnx")
        if name == "denoise":
            model, kind = build_denoiser(sd)
            inputs, names = [image], ["input"]
        elif name == "anicc":
            model, kind = build_anicc(sd)
            # The LR picture is smaller than the HR one, as in a dataset.
            lr = torch.nn.functional.interpolate(image, scale_factor=0.5, mode="bilinear", antialias=True)
            inputs, names = [image, lr], ["hr", "lr"]
        else:
            model, kind = build_prism(sd)
            strength = torch.full_like(image[:, :1], 20.0 / 51.0)
            inputs, names = [torch.cat([image, strength], dim=1)], ["input"]
        print(f"{name}: loaded as {kind}")
        export(name, model, inputs, names, out)


if __name__ == "__main__":
    main()
