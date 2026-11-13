#!/usr/bin/env python
"""Export the official ResNet50 MixVPR (4096-d) to ONNX for TensorRT 10.8.

The aggregator matches the official MixVPR release. The backbone is a ResNet50
with the last residual stage removed, so a 320x320 image becomes a 1024x20x20 map.
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

    def forward(self, x):
        return x + self.mix(x)


class MixVPRHead(nn.Module):
    def __init__(self, in_channels=1024, in_h=20, in_w=20, out_channels=1024,
                 mix_depth=4, mlp_ratio=1, out_rows=4):
        super().__init__()
        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=-1)


class ResNetEncoder(nn.Module):
    """ResNet50 cropped before layer4. Output is 1024x20x20 for a 320x320 input."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.avgpool = None
        model.fc = None
        model.layer4 = None
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


class VPRModel(nn.Module):
    def __init__(self, out_channels=1024, out_rows=4):
        super().__init__()
        self.backbone = ResNetEncoder()
        self.aggregator = MixVPRHead(out_channels=out_channels, out_rows=out_rows)

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def disable_inplace(module):
    for child in module.modules():
        if isinstance(child, (nn.ReLU, nn.ReLU6)) and hasattr(child, "inplace"):
            child.inplace = False


def checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    return cleaned


def aggregator_shape(state):
    out_channels = int(state["aggregator.channel_proj.weight"].shape[0])
    out_rows = int(state["aggregator.row_proj.weight"].shape[0])
    return out_channels, out_rows


def load_rgb(path, width, height, device):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(STD, device=device).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def save_descriptor(path, descriptor):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor.detach().float().cpu().numpy().tofile(path)


def export_onnx(module, args, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["descriptor"],
    )
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, args, str(path), **kwargs)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image0.png"),
    )
    parser.add_argument(
        "--image2",
        type=Path,
        default=Path("/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image1.png"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = checkpoint_state(args.checkpoint)
    out_channels, out_rows = aggregator_shape(state)
    print(f"aggregator out_channels={out_channels} out_rows={out_rows} dim={out_channels * out_rows}")
    model = VPRModel(out_channels=out_channels, out_rows=out_rows).to(device).eval()
    disable_inplace(model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    if real_missing:
        raise RuntimeError(f"checkpoint is missing {len(real_missing)} tensors, first: {real_missing[:8]}")
    print(f"loaded {args.checkpoint.name}: missing={len(missing)} unexpected={len(unexpected)}")
    image0 = load_rgb(args.image, args.width, args.height, device)
    image1 = load_rgb(args.image2, args.width, args.height, device)
    with torch.no_grad():
        desc0 = model(image0)
        desc1 = model(image1)
    cosine = torch.nn.functional.cosine_similarity(desc0, desc1).item()
    print(
        f"pytorch dim={desc0.shape[-1]} norm={desc0.norm().item():.6f} "
        f"cosine={cosine:.6f}"
    )
    reference_dir = args.output.parent
    tag = f"{desc0.shape[-1]}"
    save_descriptor(reference_dir / f"pytorch_{tag}_image0.bin", desc0)
    save_descriptor(reference_dir / f"pytorch_{tag}_image1.bin", desc1)
    export_onnx(model, (image0,), args.output)


if __name__ == "__main__":
    sys.exit(main())
