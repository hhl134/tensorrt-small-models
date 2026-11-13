#!/usr/bin/env python
"""Export official YOLOv8n to ONNX and to the tensorrtx .wts weight list."""

import argparse
import struct
from pathlib import Path

import torch


def export_onnx(weights, output):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        output.write_bytes(exported.read_bytes())
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def export_wts(weights, output):
    model = torch.load(weights, map_location="cpu", weights_only=False)["model"].float().eval()
    head = model.model[-1]
    if hasattr(head, "anchors"):
        delattr(head, "anchors")
    state = model.state_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write(f"{len(state)}\n")
        for key, value in state.items():
            values = value.reshape(-1).detach().cpu().numpy()
            handle.write(f"{key} {len(values)} ")
            for item in values:
                handle.write(" ")
                handle.write(struct.pack(">f", float(item)).hex())
            handle.write("\n")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, tensors={len(state)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--wts", type=Path, required=True)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-wts", action="store_true")
    args = parser.parse_args()
    if not args.skip_onnx:
        export_onnx(args.weights, args.onnx)
    if not args.skip_wts:
        export_wts(args.weights, args.wts)


if __name__ == "__main__":
    raise SystemExit(main())
