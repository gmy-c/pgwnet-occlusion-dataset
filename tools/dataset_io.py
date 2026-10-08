"""Read the author's COCO annotations and occlusion manifests without rewriting them."""
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
from pycocotools import mask as mask_utils

LEVELS = ("L", "M", "H")
BOUNDS = {"L": (0.10, 0.25), "M": (0.25, 0.45), "H": (0.45, 0.65)}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(segmentation, height, width):
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    if mask.shape != (height, width):
        raise ValueError(f"Mask shape {mask.shape} differs from {(height, width)}")
    return mask.astype(bool)


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def tight_box(mask):
    return mask_utils.toBbox(encode(mask)).tolist()


def mask_iou(first, second):
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def box_iou(first, second):
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0.0, min(x1+w1, x2+w2)-max(x1, x2)) * max(0.0, min(y1+h1, y2+h2)-max(y1, y2))
    union = w1*h1 + w2*h2 - intersection
    return intersection / union if union > 0 else 0.0


def boundary_band(mask):
    height, width = mask.shape
    radius = max(1, round(0.005 * math.hypot(height, width)))
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~eroded.astype(bool)


def boundary_iou(first, second):
    return mask_iou(boundary_band(first), boundary_band(second))


def set_name(split, level):
    return f"val_occlusion_{split}_{level}"


def dataset(root, split, level):
    return load_json(Path(root) / "annotations" / (set_name(split, level) + ".json"))


def manifest(root, split, level="L"):
    return load_json(Path(root) / "manifests" / (set_name(split, level) + ".json"))


def image_path(root, split, level, recorded):
    # Prefer the local tree so an archive moved from the author's server stays usable.
    local = Path(root) / "images" / set_name(split, level) / Path(recorded).name
    if local.is_file():
        return local
    path = Path(recorded)
    if path.is_file():
        return path
    relative = Path(root) / recorded
    if relative.is_file():
        return relative
    raise FileNotFoundError(local)


def index_annotations(data):
    images = {image["id"]: image for image in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate image IDs")
    by_image = {image_id: [] for image_id in images}
    seen = set()
    for ann in data["annotations"]:
        if ann["id"] in seen or ann["image_id"] not in images:
            raise ValueError("Duplicate annotation ID or unknown image ID")
        seen.add(ann["id"])
        by_image[ann["image_id"]].append(ann)
    return images, by_image
