#!/usr/bin/env python
"""Recall@K for MixVPR. PyTorch and TensorRT share the OpenCV 320x320 preprocess.

Images follow the VPR-datasets-downloader layout. A filename is
@utm_east@utm_north@...@.jpg. A database image is a positive when it is within
25 metres. Queries with no positive are left out of the recall.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from export_mixvpr import MEAN, STD, VPRModel, aggregator_shape, checkpoint_state, disable_inplace

ROOT = Path(__file__).resolve().parents[1]


def frame_id(path):
    return int(path.name.split("@")[7])


def collect_split(image_root, limit):
    database = sorted((image_root / "database").glob("*.jpg"), key=frame_id)
    queries = sorted((image_root / "queries").glob("*.jpg"), key=frame_id)
    query_by_frame = {frame_id(path): path for path in queries}
    paired_db = []
    paired_q = []
    for path in database:
        query = query_by_frame.get(frame_id(path))
        if query is None:
            continue
        paired_db.append(path)
        paired_q.append(query)
    if limit > 0:
        paired_db = paired_db[:limit]
        paired_q = paired_q[:limit]
    if not paired_db:
        raise FileNotFoundError(f"no paired images under {image_root}")
    return paired_db, paired_q


def utm(path):
    parts = path.name.split("@")
    return float(parts[1]), float(parts[2])


def positives(database, queries, radius):
    db = np.array([utm(path) for path in database], dtype=np.float64)
    query = np.array([utm(path) for path in queries], dtype=np.float64)
    # Coordinates are two-dimensional and the sets are a few tens of thousands.
    found = []
    for start in range(0, len(query), 512):
        block = query[start:start + 512]
        distance = np.sqrt(((block[:, None, :] - db[None, :, :]) ** 2).sum(axis=2))
        found.extend(np.flatnonzero(row <= radius) for row in distance)
    return found


def preprocess(path, width, height):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def extract_pytorch(checkpoint, paths, cache, batch_size, width, height):
    if cache.exists():
        desc = np.load(cache)
        if desc.shape[0] == len(paths):
            return desc
    device = torch.device("cuda")
    state = checkpoint_state(checkpoint)
    out_channels, out_rows = aggregator_shape(state)
    model = VPRModel(out_channels=out_channels, out_rows=out_rows).to(device).eval()
    disable_inplace(model)
    missing, _ = model.load_state_dict(state, strict=False)
    real_missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    if real_missing:
        raise RuntimeError(real_missing[:8])
    chunks = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([
                preprocess(path, width, height) for path in paths[start:start + batch_size]
            ]).to(device)
            chunks.append(model(batch).float().cpu().numpy())
            if start == 0 or (start // batch_size) % 20 == 0:
                print(f"pytorch {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    desc = np.concatenate(chunks, axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, desc)
    return desc


def extract_tensorrt(config, paths, cache, list_path, dim):
    expected = len(paths) * dim
    if cache.exists() and cache.stat().st_size == expected * 4:
        return np.fromfile(cache, dtype=np.float32)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("".join(f"{path}\n" for path in paths))
    binary = ROOT / "build" / "mixvpr_extract"
    subprocess.run(
        [str(binary), str(config), str(ROOT / "weights"), str(list_path), str(cache)],
        check=True,
    )
    return np.fromfile(cache, dtype=np.float32)


def recall_at(database, queries, positive_sets, ks):
    database_t = torch.from_numpy(database).cuda()
    queries_t = torch.from_numpy(queries).cuda()
    database_t = torch.nn.functional.normalize(database_t, dim=1)
    queries_t = torch.nn.functional.normalize(queries_t, dim=1)
    hits = {k: 0 for k in ks}
    valid = 0
    max_k = max(ks)
    step = 512
    for start in range(0, queries_t.shape[0], step):
        similarity = queries_t[start:start + step] @ database_t.T
        index = torch.topk(similarity, k=min(max_k, database_t.shape[0]), dim=1).indices.cpu().numpy()
        for row, query_index in enumerate(range(start, start + index.shape[0])):
            truth = positive_sets[query_index]
            if truth.size == 0:
                continue
            valid += 1
            ranked = index[row]
            for k in ks:
                if np.intersect1d(ranked[:k], truth).size:
                    hits[k] += 1
    if valid == 0:
        raise RuntimeError("no query has a positive")
    return valid, {k: hits[k] / valid for k in ks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--radius", type=float, default=25.0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "out" / "recall")
    args = parser.parse_args()

    database_paths, query_paths = collect_split(args.image_root, args.limit)
    frames = [frame_id(path) for path in database_paths]
    gaps = sum(b != a + 1 for a, b in zip(frames, frames[1:]))
    print(
        f"database={len(database_paths)} queries={len(query_paths)} "
        f"frames={frames[0]}-{frames[-1]} gaps={gaps}",
        flush=True,
    )
    positive_sets = positives(database_paths, query_paths, args.radius)
    covered = sum(item.size > 0 for item in positive_sets)
    print(f"queries with a positive={covered}/{len(query_paths)}", flush=True)

    runs = [
        ("pytorch_4096", ROOT / "weights" / "resnet50_MixVPR_4096.ckpt", None, 4096),
        ("trt_4096_fp16", None, ROOT / "config" / "config.yaml", 4096),
        ("trt_4096_fp32", None, ROOT / "config" / "bench" / "4096_fp32.yaml", 4096),
        ("trt_4096_fp16_all", None, ROOT / "config" / "bench" / "4096_fp16_all.yaml", 4096),
        ("pytorch_512", ROOT / "weights" / "resnet50_MixVPR_512.ckpt", None, 512),
        ("trt_512_fp16", None, ROOT / "config" / "bench" / "512_fp16.yaml", 512),
        ("pytorch_128", ROOT / "weights" / "resnet50_MixVPR_128.ckpt", None, 128),
        ("trt_128_fp16", None, ROOT / "config" / "bench" / "128_fp16.yaml", 128),
    ]
    tag = f"n{len(database_paths)}"
    descriptors = {}
    for name, checkpoint, config, dim in runs:
        cache = args.cache_dir / f"{name}_{tag}.npy" if checkpoint else args.cache_dir / f"{name}_{tag}.bin"
        print(f"extract {name}", flush=True)
        if checkpoint:
            db = extract_pytorch(checkpoint, database_paths, cache.with_name(cache.stem + "_db.npy"), args.batch_size, 320, 320)
            query = extract_pytorch(checkpoint, query_paths, cache.with_name(cache.stem + "_q.npy"), args.batch_size, 320, 320)
        else:
            db_bin = cache.with_name(cache.stem + "_db.bin")
            q_bin = cache.with_name(cache.stem + "_q.bin")
            db = extract_tensorrt(config, database_paths, db_bin, args.cache_dir / f"{name}_{tag}_db.txt", dim).reshape(-1, dim)
            query = extract_tensorrt(config, query_paths, q_bin, args.cache_dir / f"{name}_{tag}_q.txt", dim).reshape(-1, dim)
        descriptors[name] = (db, query)
        valid, scores = recall_at(db, query, positive_sets, (1, 5, 10))
        print(
            f"{name} valid={valid} R@1={scores[1] * 100:.2f} "
            f"R@5={scores[5] * 100:.2f} R@10={scores[10] * 100:.2f}",
            flush=True,
        )

    reference = descriptors["pytorch_4096"][0]
    for name in ("trt_4096_fp16", "trt_4096_fp32", "trt_4096_fp16_all"):
        other = descriptors[name][0]
        cosine = np.sum(reference * other, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(other, axis=1)
        )
        print(
            f"cosine {name} vs pytorch_4096 mean={cosine.mean():.6f} min={cosine.min():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
