"""Audit paired COCO data, visible annotations, hashes and optional source-pixel replay."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dataset_io import (BOUNDS, LEVELS, dataset, decode, image_path, index_annotations,
                        load_json, manifest, sha256, tight_box, write_json)

REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def in_severity(ratio, level):
    low, high = BOUNDS[level]
    return low <= ratio <= high if level == "H" else low <= ratio < high


def audit(root, source_ann=None, source_images=None, require_paper_counts=False):
    root = Path(root)
    require((source_ann is None) == (source_images is None), "Supply both --source-ann and --source-images")
    source_data = load_json(source_ann) if source_ann is not None else None
    source_index = index_annotations(source_data) if source_data is not None else None
    partitions = load_json(root / "partitions.json")
    parts = partitions["legacy_source_partitions"]
    pools = partitions["donor_pools"]
    for first, second in [("dev", "locked_test"), ("dev", "donor_val"), ("locked_test", "donor_val")]:
        require(not set(parts[first]) & set(parts[second]), f"Source partition overlap: {first}/{second}")
    require(not set(pools["dev"]) & set(pools["report"]), "Development/report donor overlap")
    require(set(pools["dev"]) | set(pools["report"]) == set(parts["donor_val"]), "Donor pools differ from the frozen donor partition")
    supplied = load_json(REPO / "generation_code/partitions.json")
    if require_paper_counts:
        require(partitions == supplied, "Use the published frozen partitions for paper reproduction")
    summary = {"status": "passed", "standard": "manuscript Section 4.4; H includes 0.65",
               "source_pixel_replay": source_data is not None, "sets": {},
               "generated_images_checked": 0, "original_instances_checked": 0,
               "pixel_compositions_checked": 0}
    accepted = {}
    donor_sets = {}
    for split, expected in [("dev", 60), ("report", 300)]:
        rows = {level: manifest(root, split, level) for level in LEVELS}
        maps = {level: {(row["source_image_id"], row["target_ann_id"]): row for row in records}
                for level, records in rows.items()}
        require(all(len(maps[level]) == len(rows[level]) for level in LEVELS), "Duplicate source-target records")
        require(bool(maps["L"]) and all(set(maps[level]) == set(maps["L"]) for level in LEVELS), "Cross-level target mismatch")
        require(len({key[0] for key in maps["L"]}) == len(maps["L"]), "Expected one target per source image")
        if require_paper_counts:
            require(len(maps["L"]) == expected, f"Expected {expected} {split} targets")
        accepted[split] = {key[0] for key in maps["L"]}
        donor_sets[split] = {row["donor_image_id"] for row in rows["L"]}
        require(accepted[split] <= set(parts["dev" if split == "dev" else "locked_test"]), "Target outside its source partition")
        require(donor_sets[split] <= set(pools[split]), "Donor outside its frozen pool")
        clean = dataset(root, split, "clean")
        clean_images, clean_anns = index_annotations(clean)
        require(set(clean_images) == accepted[split], "Clean images do not match paired targets")
        by_level = {}
        for level in LEVELS:
            data = dataset(root, split, level)
            require(data["categories"] == clean["categories"], "Category mapping changed between views")
            images, annotations = index_annotations(data)
            require(set(images) == accepted[split], "Generated image IDs differ from clean")
            by_level[level] = images, annotations
        class_counts = Counter()
        sizes = Counter()
        for key in maps["L"]:
            triple = [maps[level][key] for level in LEVELS]
            identity_fields = ("source_image_id", "target_ann_id", "target_category_id", "donor_image_id", "donor_ann_id", "scale", "y")
            require(len({tuple(row[field] for field in identity_fields) for row in triple}) == 1, "Donor identity, scale or vertical placement changed")
            ratios = [row["added_occlusion_ratio"] for row in triple]
            require(ratios[0] < ratios[1] < ratios[2], "Severity must increase across L/M/H")
            require(triple[0]["x"] < triple[1]["x"] < triple[2]["x"], "Expected the fixed left-entry trajectory")
            image_id, target_id = key
            clean_image = clean_images[image_id]
            height, width = clean_image["height"], clean_image["width"]
            if require_paper_counts:
                require((height, width) == (720, 1280), "Paper resolution differs")
            original_anns = clean_anns[image_id]
            original = {ann["id"]: ann for ann in original_anns}
            require(target_id in original, "Original target annotation is missing")
            target = decode(original[target_id]["segmentation"], height, width)
            area = int(target.sum())
            require(area > 0, "Empty original target")
            class_counts[original[target_id]["category_id"]] += 1
            sizes["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
            clean_path = image_path(root, split, "clean", clean_image["file_name"])
            clean_pixels = cv2.imread(str(clean_path))
            require(clean_pixels is not None and clean_pixels.shape[:2] == (height, width), "Invalid clean pixels")
            if source_data is not None:
                source_im = source_index[0][image_id]
                require(source_index[1][image_id] == original_anns, "Clean annotations differ from source COCO annotations")
                source_path = Path(source_images) / Path(source_im["file_name"]).name
                require(sha256(source_path) == sha256(clean_path), "Clean copy differs from source image bytes")
            for level, row in zip(LEVELS, triple):
                images, annotations = by_level[level]
                im = images[image_id]
                require((im["height"], im["width"]) == (height, width), "Image dimensions changed")
                visible = decode(row["original_visible_target"], height, width)
                require(np.array_equal(visible, target), "Saved original visible target differs from clean")
                occ = decode(row["occluder"], height, width)
                ratio = float(np.count_nonzero(target & occ) / area)
                require(abs(ratio-row["added_occlusion_ratio"]) < 1e-10 and in_severity(ratio, level), "Realized severity mismatch")
                require(np.count_nonzero(target & ~occ) >= 64, "Target falls below the constructor's retained-area threshold")
                mapping = {ann["source_ann_id"]: ann for ann in annotations[image_id]}
                require(len(mapping) == len(annotations[image_id]) and -1 in mapping, "Invalid source annotation mapping or missing donor")
                require(target_id in mapping and mapping[target_id]["id"] == row["generated_target_ann_id"], "Generated target identity mismatch")
                expected_ids = {-1}
                expected_changes = []
                for ann in original_anns:
                    old = decode(ann["segmentation"], height, width)
                    remaining = old & ~occ
                    expected_changes.append({"source_ann_id": ann["id"], "old_area": int(old.sum()), "new_area": int(remaining.sum())})
                    summary["original_instances_checked"] += 1
                    if not remaining.any():
                        require(ann["id"] not in mapping, "Fully hidden instance was retained")
                        continue
                    expected_ids.add(ann["id"])
                    actual = mapping[ann["id"]]
                    require(actual["category_id"] == ann["category_id"], "Original instance class changed")
                    require(np.array_equal(decode(actual["segmentation"], height, width), remaining), "Visible-mask subtraction mismatch")
                    require(actual["area"] == int(remaining.sum()) and np.allclose(actual["bbox"], tight_box(remaining)), "Incorrect updated area or box")
                require(row["changes"] == expected_changes, "Recorded instance area changes differ from masks")
                require(set(mapping) == expected_ids, "Unexpected or missing scene instance")
                donor = mapping[-1]
                require(np.array_equal(decode(donor["segmentation"], height, width), occ), "Donor mask mismatch")
                require(donor["area"] == int(occ.sum()) and np.allclose(donor["bbox"], tight_box(occ)), "Incorrect donor area or box")
                path = image_path(root, split, level, row["generated_file"])
                require(sha256(path) == row.get("image_sha256", row.get("sha256")), "Generated image hash mismatch")
                pixels = cv2.imread(str(path))
                require(pixels is not None and pixels.shape[:2] == (height, width), "Invalid generated pixels")
                require(np.array_equal(clean_pixels[~occ], pixels[~occ]), "Pixels outside the donor changed")
                summary["generated_images_checked"] += 1
                if source_data is not None:
                    donor_anns = source_index[1][row["donor_image_id"]]
                    donor_ann = next(ann for ann in donor_anns if ann["id"] == row["donor_ann_id"])
                    require(donor["category_id"] == donor_ann["category_id"], "Donor category mismatch")
                    donor_im = source_index[0][row["donor_image_id"]]
                    dm = decode(donor_ann["segmentation"], donor_im["height"], donor_im["width"])
                    yy, xx = np.where(dm)
                    donor_path = Path(source_images) / Path(donor_im["file_name"]).name
                    donor_pixels = cv2.imread(str(donor_path))
                    require(donor_pixels is not None, "Missing donor source pixels")
                    crop = donor_pixels[yy.min():yy.max()+1, xx.min():xx.max()+1]
                    cut = dm[yy.min():yy.max()+1, xx.min():xx.max()+1]
                    dh, dw = max(2, round(cut.shape[0]*row["scale"])), max(2, round(cut.shape[1]*row["scale"]))
                    resized_mask = cv2.resize(cut.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
                    resized_pixels = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
                    x, y = row["x"], row["y"]
                    reconstructed = np.zeros_like(occ)
                    reconstructed[y:y+dh, x:x+dw] = resized_mask
                    require(np.array_equal(reconstructed, occ), "Recorded donor transformation differs from mask")
                    require(np.array_equal(pixels[y:y+dh, x:x+dw][resized_mask], resized_pixels[resized_mask]), "Donor pixels differ from the recorded source transformation")
                    summary["pixel_compositions_checked"] += 1
        category_names = {category["id"]: category["name"] for category in clean["categories"]}
        distribution = {category_names[category]: count for category, count in class_counts.items()}
        if require_paper_counts and split == "report":
            paper = {"pedestrian": 64, "rider": 17, "car": 61, "truck": 58, "bus": 56, "train": 2, "motorcycle": 22, "bicycle": 20}
            require(distribution == paper, "Report category counts differ from the manuscript")
            require(dict(sizes) == {"small": 136, "medium": 115, "large": 49}, "Report size counts differ from the manuscript")
        summary["sets"][split] = {"targets": len(maps["L"]), "target_views": 4*len(maps["L"]),
                                 "categories": distribution, "original_visible_size": dict(sizes),
                                 "ratio_ranges": {level: [min(row["added_occlusion_ratio"] for row in rows[level]),
                                                          max(row["added_occlusion_ratio"] for row in rows[level])] for level in LEVELS}}
    require(not accepted["dev"] & accepted["report"], "Accepted source image overlap")
    require(not donor_sets["dev"] & donor_sets["report"], "Accepted donor pool overlap")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-ann", type=Path)
    parser.add_argument("--source-images", type=Path)
    parser.add_argument("--require-paper-counts", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.source_ann, args.source_images, args.require_paper_counts)
    write_json(args.out, result)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
