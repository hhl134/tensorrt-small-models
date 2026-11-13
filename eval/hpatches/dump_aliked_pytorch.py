#!/usr/bin/env python3
"""Dump ALIKED+LightGlue PyTorch matches on HPatches at the TensorRT input size."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/hhl/workspace/algorithm/LightGlue")
torch.hub.set_dir("/home/hhl/workspace/deploy/tensorrt/ALIKED-LightGlue-TensorRT/weights/torchhub")

from lightglue import ALIKED, LightGlue  # noqa: E402

WIDTH = 640
HEIGHT = 480
ROOT = Path(sys.argv[1])
VARIANT = sys.argv[2]
OUTPUT = Path(sys.argv[3])
MAX_SEQ = int(os.environ.get("HP_MAX_SEQ", "0"))


def build_models():
    if VARIANT == "pytorch_default":
        extractor = ALIKED(max_num_keypoints=-1, detection_threshold=0.2)
        matcher = LightGlue(features="aliked")
    elif VARIANT == "pytorch_k256_dynamic":
        extractor = ALIKED(max_num_keypoints=256, detection_threshold=-1)
        matcher = LightGlue(features="aliked")
    elif VARIANT == "pytorch_threshold_static":
        extractor = ALIKED(max_num_keypoints=-1, detection_threshold=0.2)
        matcher = LightGlue(
            features="aliked",
            flash=False,
            mp=False,
            depth_confidence=-1,
            width_confidence=-1,
        )
    elif VARIANT == "pytorch_k256_static":
        extractor = ALIKED(max_num_keypoints=256, detection_threshold=-1)
        matcher = LightGlue(
            features="aliked",
            flash=False,
            mp=False,
            depth_confidence=-1,
            width_confidence=-1,
        )
    else:
        raise SystemExit(f"Unknown variant {VARIANT}")
    return extractor.eval().cuda(), matcher.eval().cuda()


def load_resized(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    original = (image.shape[1], image.shape[0])
    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.cuda(), original


def main():
    extractor, matcher = build_models()
    image_size = torch.tensor([[WIDTH, HEIGHT]], device="cuda", dtype=torch.float32)
    sequences = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    if MAX_SEQ:
        sequences = sequences[:MAX_SEQ]
    written = 0
    with OUTPUT.open("w") as handle:
        for index, name in enumerate(sequences, 1):
            image0, size0 = load_resized(ROOT / name / "1.ppm")
            if image0 is None:
                continue
            with torch.no_grad():
                feats0 = extractor({"image": image0})
            for pair in range(2, 7):
                image1, size1 = load_resized(ROOT / name / f"{pair}.ppm")
                if image1 is None:
                    continue
                with torch.no_grad():
                    feats1 = extractor({"image": image1})
                    pred = matcher({
                        "image0": {**feats0, "image_size": image_size},
                        "image1": {**feats1, "image_size": image_size},
                    })
                matches = pred["matches"][0].detach().cpu().numpy()
                kpts0 = feats0["keypoints"][0].detach().cpu().numpy()
                kpts1 = feats1["keypoints"][0].detach().cpu().numpy()
                sx0, sy0 = size0[0] / WIDTH, size0[1] / HEIGHT
                sx1, sy1 = size1[0] / WIDTH, size1[1] / HEIGHT
                kpts0 = kpts0 * np.array([sx0, sy0], dtype=np.float32)
                kpts1 = kpts1 * np.array([sx1, sy1], dtype=np.float32)
                picked0 = kpts0[matches[:, 0]] if len(matches) else np.zeros((0, 2))
                picked1 = kpts1[matches[:, 1]] if len(matches) else np.zeros((0, 2))
                values = [name, str(pair), str(size0[0]), str(size0[1]), str(len(matches))]
                for p0, p1 in zip(picked0, picked1):
                    values.extend(f"{v:.3f}" for v in (p0[0], p0[1], p1[0], p1[1]))
                handle.write(" ".join(values) + "\n")
                written += 1
            if index % 10 == 0:
                print(f"sequences {index} pairs {written}", file=sys.stderr, flush=True)
    print(f"Wrote {written} pairs to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
