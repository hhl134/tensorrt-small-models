#!/usr/bin/env python3
"""Score dumped HPatches matches by homography corner error."""

import sys
from pathlib import Path

import cv2
import numpy as np


def corner_error(homography, homography_gt, width, height):
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]],
        dtype=np.float64,
    ).T

    def project(matrix):
        warped = matrix @ corners
        return (warped[:2] / warped[2]).T

    return float(np.linalg.norm(project(homography) - project(homography_gt), axis=1).mean())


def auc_at(errors, thresholds):
    ordered = np.sort(np.asarray(errors, dtype=np.float64))
    recall = (np.arange(len(ordered)) + 1) / len(ordered)
    ordered = np.r_[0.0, ordered]
    recall = np.r_[0.0, recall]
    aucs = []
    for threshold in thresholds:
        last = np.searchsorted(ordered, threshold)
        recall_curve = np.r_[recall[:last], recall[last - 1]]
        error_curve = np.r_[ordered[:last], threshold]
        trap = getattr(np, "trapezoid", np.trapz)
        aucs.append(float(trap(recall_curve, error_curve) / threshold))
    return aucs


def load_gt(root, name, pair):
    text = Path(root, name, f"H_1_{pair}").read_text().split()
    return np.array([float(v) for v in text], dtype=np.float64).reshape(3, 3)


def score_file(path, root):
    errors = []
    tags = []
    match_counts = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        parts = line.split()
        name, pair = parts[0], int(parts[1])
        width, height, count = int(parts[2]), int(parts[3]), int(parts[4])
        nums = np.array([float(v) for v in parts[5:]], dtype=np.float32)
        error = np.inf
        if count >= 4:
            pts = nums.reshape(-1, 4)
            cv2.setRNGSeed(0)
            homography, _ = cv2.findHomography(
                pts[:, :2], pts[:, 2:], cv2.RANSAC, 3.0, None, 5000, 0.999
            )
            if homography is not None and np.isfinite(homography).all():
                try:
                    error = corner_error(homography, load_gt(root, name, pair), width, height)
                except np.linalg.LinAlgError:
                    error = np.inf
                if not np.isfinite(error):
                    error = np.inf
        errors.append(error)
        tags.append(name[0])
        match_counts.append(count)
    return np.array(errors), np.array(tags), np.array(match_counts)


def summarize(errors, tags, match_counts):
    finite = errors[np.isfinite(errors)]
    auc1, auc3, auc5 = auc_at(errors, (1.0, 3.0, 5.0))
    row = {
        "pairs": int(len(errors)),
        "matches": float(match_counts.mean()) if len(match_counts) else 0.0,
        "acc1": float(np.mean(errors < 1)),
        "acc3": float(np.mean(errors < 3)),
        "acc5": float(np.mean(errors < 5)),
        "auc1": auc1,
        "auc3": auc3,
        "auc5": auc5,
        "median": float(np.median(finite)) if len(finite) else float("inf"),
    }
    splits = {}
    for tag, label in (("v", "viewpoint"), ("i", "illumination")):
        mask = tags == tag
        if not np.any(mask):
            continue
        part = errors[mask]
        part_auc = auc_at(part, (1.0, 3.0, 5.0))
        splits[label] = {
            "pairs": int(mask.sum()),
            "acc1": float(np.mean(part < 1)),
            "acc3": float(np.mean(part < 3)),
            "acc5": float(np.mean(part < 5)),
            "auc1": part_auc[0],
            "auc3": part_auc[1],
            "auc5": part_auc[2],
            "matches": float(match_counts[mask].mean()),
        }
    row["splits"] = splits
    return row


def format_pct(value):
    return f"{100.0 * value:.1f}"


def main():
    root = sys.argv[1]
    for path in sys.argv[2:]:
        errors, tags, match_counts = score_file(path, root)
        row = summarize(errors, tags, match_counts)
        print(
            f"{Path(path).stem}\tpairs={row['pairs']}\tmatches={row['matches']:.1f}"
            f"\tacc@1/3/5={format_pct(row['acc1'])}/{format_pct(row['acc3'])}/{format_pct(row['acc5'])}"
            f"\tauc@1/3/5={format_pct(row['auc1'])}/{format_pct(row['auc3'])}/{format_pct(row['auc5'])}"
            f"\tmedian={row['median']:.2f}"
        )
        for label, split in row["splits"].items():
            print(
                f"  {label}\tacc@1/3/5={format_pct(split['acc1'])}/{format_pct(split['acc3'])}/{format_pct(split['acc5'])}"
                f"\tauc@1/3/5={format_pct(split['auc1'])}/{format_pct(split['auc3'])}/{format_pct(split['auc5'])}"
                f"\tmatches={split['matches']:.1f}"
            )


if __name__ == "__main__":
    main()
