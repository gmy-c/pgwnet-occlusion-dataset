"""Target metrics from manuscript Section 4.4; input predictions are already postprocessed."""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from dataset_io import (boundary_iou, box_iou, dataset, decode, index_annotations,
                        load_json, manifest, mask_iou, tight_box, write_json)

THRESHOLDS = tuple(round(0.50 + 0.05*k, 2) for k in range(10))


def associate(predictions, ground_truth, similarities, threshold):
    """Score order is preserved; sorted GT IDs implement deterministic IoU ties."""
    matched = {}
    used = set()
    for prediction_index, pred in enumerate(predictions):
        candidates = [index for index, ann in enumerate(ground_truth)
                      if index not in used and ann["category_id"] == pred["category_id"]]
        if not candidates:
            continue
        best = max(candidates, key=lambda index: (similarities[prediction_index, index],
                                                  -ground_truth[index]["id"]))
        if similarities[prediction_index, best] >= threshold:
            used.add(best)
            matched[ground_truth[best]["id"]] = prediction_index
    return matched


def score_image(image, annotations, predictions, target_id):
    height, width = image["height"], image["width"]
    annotations = sorted(annotations, key=lambda ann: ann["id"])
    if target_id not in {ann["id"] for ann in annotations}:
        raise ValueError(f"Missing designated target {target_id} in image {image['id']}")
    if len(predictions) > 100:
        raise ValueError("Prediction input exceeds the manuscript's 100-instance output limit")
    for pred in predictions:
        if not math.isfinite(pred["score"]) or not 0 <= pred["score"] <= 1:
            raise ValueError("Prediction scores must be finite values in [0, 1]")
        if len(pred.get("bbox", [])) != 4 or not all(math.isfinite(v) for v in pred["bbox"]):
            raise ValueError("Prediction bbox must be a finite COCO [x, y, width, height] box")
        if pred["bbox"][2] < 0 or pred["bbox"][3] < 0:
            raise ValueError("Prediction box dimensions must be nonnegative")
    # Stable sort retains the original JSON output order for equal scores.
    predictions = sorted((pred for pred in predictions if pred["score"] >= 0.30),
                         key=lambda pred: -pred["score"])
    gt_masks = [decode(ann["segmentation"], height, width) for ann in annotations]
    pred_masks = [decode(pred["segmentation"], height, width) for pred in predictions]
    gt_boxes = [tight_box(mask) for mask in gt_masks]
    overlaps = np.array([[mask_iou(pred, gt) for gt in gt_masks] for pred in pred_masks], dtype=float)
    overlaps = overlaps.reshape(len(predictions), len(annotations))
    box_overlaps = np.array([[box_iou(pred["bbox"], box) for box in gt_boxes]
                            for pred in predictions], dtype=float).reshape(len(predictions), len(annotations))
    recovered = {str(threshold): int(target_id in associate(predictions, annotations, overlaps, threshold))
                 for threshold in THRESHOLDS}
    quality = associate(predictions, annotations, box_overlaps, 0.50)
    target_index = next(i for i, ann in enumerate(annotations) if ann["id"] == target_id)
    pred_index = quality.get(target_id)
    return {"image_id": image["id"], "target_ann_id": target_id, "recovered": recovered,
            "mask_iou": float(overlaps[pred_index, target_index]) if pred_index is not None else 0.0,
            "boundary_iou": boundary_iou(pred_masks[pred_index], gt_masks[target_index]) if pred_index is not None else 0.0,
            "quality_prediction_rank": pred_index}


def evaluate(root, split, level, predictions_path):
    data = dataset(root, split, level)
    images, by_image = index_annotations(data)
    rows = manifest(root, split, "L" if level == "clean" else level)
    targets = {row["source_image_id"]: row["target_ann_id"] if level == "clean"
               else row["generated_target_ann_id"] for row in rows}
    if not targets or len(targets) != len(rows) or set(targets) != set(images):
        raise ValueError("Expected one designated target per paired image")
    predictions = load_json(predictions_path)
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO-style JSON list")
    grouped = defaultdict(list)
    for pred in predictions:
        if pred["image_id"] not in images:
            raise ValueError(f"Unknown prediction image ID: {pred['image_id']}")
        grouped[pred["image_id"]].append(pred)
    scores = [score_image(images[i], by_image[i], grouped[i], target_id) for i, target_id in targets.items()]
    recall = {threshold: 100*sum(row["recovered"][str(threshold)] for row in scores)/len(scores)
              for threshold in THRESHOLDS}
    return {"split": split, "level": level, "targets": len(scores), "scale": "0-100",
            "metrics": {"R50": recall[0.50], "R75": recall[0.75],
                        "R50:95": sum(recall.values())/10,
                        "Mask mIoU": 100*sum(row["mask_iou"] for row in scores)/len(scores),
                        "Boundary mIoU": 100*sum(row["boundary_iou"] for row in scores)/len(scores)},
            "implementation": "Repository implementation of manuscript Section 4.4; not the supplied historical AP runner",
            "per_target": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "report"], default="report")
    parser.add_argument("--level", choices=["clean", "L", "M", "H"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root, args.split, args.level, args.predictions)
    write_json(args.out, result)
    import json
    print(json.dumps({key: result[key] for key in ["split", "level", "targets", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
