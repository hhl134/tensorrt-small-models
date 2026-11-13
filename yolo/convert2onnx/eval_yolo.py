import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def letterbox(bgr):
    height, width = bgr.shape[:2]
    scale = min(640 / width, 640 / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    pad_x = (640 - resized_w) / 2.0
    pad_y = (640 - resized_h) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    right = 640 - resized_w - left
    bottom = 640 - resized_h - top
    resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None]
    return nchw, scale, left, top, width, height


def decode(pred, scale, left, top, width, height, conf_thres, nms_thres, max_det):
    scores_all = pred[4:]
    class_id = scores_all.argmax(0)
    score = scores_all[class_id, np.arange(scores_all.shape[1])]
    keep = score >= conf_thres
    if not np.any(keep):
        return []
    cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
    score = score[keep]
    class_id = class_id[keep]
    x1 = np.clip((cx - bw * 0.5 - left) / scale, 0, width)
    y1 = np.clip((cy - bh * 0.5 - top) / scale, 0, height)
    x2 = np.clip((cx + bw * 0.5 - left) / scale, 0, width)
    y2 = np.clip((cy + bh * 0.5 - top) / scale, 0, height)
    order = np.argsort(-score)
    kept = []
    for index in order:
        box = (float(x1[index]), float(y1[index]), float(x2[index]), float(y2[index]), float(score[index]), int(class_id[index]))
        if any(box[5] == old[5] and iou(box, old) > nms_thres for old in kept):
            continue
        kept.append(box)
        if len(kept) >= max_det:
            break
    return kept


def iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - inter
    return inter / area if area > 0 else 0.0


def parse_dump(path):
    images = {}
    current = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("image "):
            current = line.split()[1]
            images[current] = []
        elif line.startswith("box ") and current is not None:
            parts = line.split()
            images[current].append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[6]), int(parts[8])))
    return images


def match_stats(left_images, right_images):
    names = sorted(set(left_images) & set(right_images))
    matched = 0
    class_agree = 0
    ious = []
    left_total = 0
    right_total = 0
    for name in names:
        left = left_images[name]
        right = right_images[name]
        left_total += len(left)
        right_total += len(right)
        used = set()
        pairs = []
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                value = iou(a, b)
                if value >= 0.5:
                    pairs.append((value, i, j))
        pairs.sort(reverse=True)
        seen_left = set()
        for value, i, j in pairs:
            if i in seen_left or j in used:
                continue
            seen_left.add(i)
            used.add(j)
            matched += 1
            ious.append(value)
            if left[i][5] == right[j][5]:
                class_agree += 1
    return {
        "images": len(names),
        "left_boxes": left_total,
        "right_boxes": right_total,
        "matched": matched,
        "class_agree": class_agree,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "left_per_image": left_total / max(len(names), 1),
        "right_per_image": right_total / max(len(names), 1),
    }


def dump_pytorch(image_dir, out_path, weights, conf, nms, max_det):
    import torch
    from ultralytics import YOLO

    net = YOLO(weights).model.cuda().eval()
    files = sorted(Path(image_dir).glob("*.jpg"))
    lines = []
    start = time.perf_counter()
    with torch.no_grad():
        for index, path in enumerate(files):
            bgr = cv2.imread(str(path))
            nchw, scale, left, top, width, height = letterbox(bgr)
            tensor = torch.from_numpy(nchw).cuda()
            pred = net(tensor)[0][0].detach().float().cpu().numpy()
            boxes = decode(pred, scale, left, top, width, height, conf, nms, max_det)
            lines.append(f"image {path.name} boxes={len(boxes)}")
            for box in boxes:
                lines.append(f"box {box[0]} {box[1]} {box[2]} {box[3]} conf {box[4]} class {box[5]}")
            if (index + 1) % 500 == 0:
                print(f"pytorch {index + 1}/{len(files)} {time.perf_counter() - start:.1f}s", flush=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} images={len(files)}", flush=True)


def to_coco(images):
    detections = []
    for name, boxes in images.items():
        image_id = int(Path(name).stem)
        for x1, y1, x2, y2, score, class_id in boxes:
            detections.append({
                "image_id": image_id,
                "category_id": COCO80_TO_91[class_id],
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": score,
            })
    return detections


def coco_map(ann_file, images):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(ann_file)
    detections = to_coco(images)
    if not detections:
        print("no detections")
        return
    result = coco.loadRes(detections)
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.maxDets = [1, 10, 100, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    # precision: [TxRxKxAxM], M follows maxDets
    for label, index in (("maxDets=100", 2), ("maxDets=300", 3)):
        all_area = precision[:, :, :, 0, index]
        ap = all_area[all_area > -1].mean()
        ap50 = precision[0, :, :, 0, index]
        ap50 = ap50[ap50 > -1].mean()
        ap75 = precision[5, :, :, 0, index]
        ap75 = ap75[ap75 > -1].mean()
        print(f"{label}  AP={ap:.4f}  AP50={ap50:.4f}  AP75={ap75:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-dir")
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--dump")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--map")
    parser.add_argument("--ann")
    args = parser.parse_args()
    if args.pytorch_dir:
        dump_pytorch(args.pytorch_dir, args.dump, args.weights, args.conf, args.nms, args.max_det)
    if args.compare:
        left = parse_dump(args.compare[0])
        right = parse_dump(args.compare[1])
        stats = match_stats(left, right)
        print(json.dumps(stats))
    if args.map:
        coco_map(args.ann, parse_dump(args.map))


if __name__ == "__main__":
    main()
