"""Synthetic protocol tests; fixtures are not samples from the released dataset."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "generation_code/original_runtime"))
from dataset_io import boundary_band, encode, manifest, sha256, tight_box, write_json
from audit_dataset import audit, in_severity
from evaluate_targets import associate, evaluate, score_image
from visualize_dataset import render
from verify_provenance import verify_bundle
from lite_operators import LEVELS, VERSION, seed_for
from build_lite import get_rgb

spec = importlib.util.spec_from_file_location("author_legacy_test", REPO / "generation_code/legacy/tools/build_protocol.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def load_triplet(source):
    # Execute the actual active adapter function, without its full 998-image CLI.
    tree = ast.parse((REPO / "generation_code/build_occlusion_only.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_triplet")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(val=source, old=legacy, get_rgb=get_rgb, seed_for=seed_for,
                 LEVELS=LEVELS, VERSION=VERSION, np=np, cv2=cv2, mu=mu, copy=copy)
    exec(compile(module, "active_adapter_triplet", "exec"), scope)
    return scope["make_triplet"]


def fixture(directory):
    root = Path(directory) / "dataset"
    sources = Path(directory) / "sources"
    root.mkdir()
    sources.mkdir()
    (root / "annotations").mkdir()
    (root / "manifests").mkdir()
    categories = [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "car"}]
    images, annotations = [], []
    height = width = 512
    for image_id in range(1, 5):
        yy, xx = np.indices((height, width))
        rgb = np.stack([(xx + image_id*13) % 256, (yy + image_id*7) % 256, (xx+yy) % 256], axis=-1).astype(np.uint8)
        file_name = f"source_{image_id}.png"
        cv2.imwrite(str(sources/file_name), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        mask = np.zeros((height, width), bool)
        if image_id <= 2:
            mask[300:400, 220:280] = True
        else:
            mask[100:200, 40:90] = True
        annotations.append({"id": image_id*100+1, "image_id": image_id,
                            "category_id": 1 if image_id <= 2 else 2,
                            "segmentation": encode(mask), "area": int(mask.sum()),
                            "bbox": tight_box(mask), "iscrowd": 0})
        if image_id <= 2:
            context = np.zeros_like(mask)
            context[350:385, 235:255] = True
            annotations.append({"id": image_id*100+2, "image_id": image_id, "category_id": 2,
                                "segmentation": encode(context), "area": int(context.sum()),
                                "bbox": tight_box(context), "iscrowd": 0})
    source_ann = Path(directory)/"source.json"
    write_json(source_ann, {"images": images, "annotations": annotations, "categories": categories})
    source = legacy.Source(source_ann, sources)
    generate = load_triplet(source)
    for split, image_id, donor_id in [("dev", 1, 3), ("report", 2, 4)]:
        anchor = (image_id, image_id*100+1, 1)
        result = generate(anchor, source.candidates([donor_id]), split)
        if result is None:
            raise AssertionError("Synthetic target did not admit a shared donor trajectory")
        for level in LEVELS:
            writer = legacy.DatasetWriter(source, root, f"val_occlusion_{split}_{level}")
            rgb, anns, row = result[level]
            writer.add(anchor, (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), anns, row))
            writer.finish()
        clean = source.export([image_id])
        for im in clean["images"]:
            src = Path(im["file_name"])
            dest = root/"images"/f"val_occlusion_{split}_clean"/src.name
            dest.parent.mkdir(parents=True)
            shutil.copy2(src, dest)
            im["file_name"] = str(dest)
        write_json(root/"annotations"/f"val_occlusion_{split}_clean.json", clean)
    write_json(root/"partitions.json", {"legacy_source_partitions": {"dev": [1], "locked_test": [2], "donor_val": [3, 4]},
                                      "donor_pools": {"dev": [3], "report": [4]}, "claim": "synthetic_test_only"})
    return root, source_ann, sources, source


class ProtocolTests(unittest.TestCase):
    def test_h_upper_endpoint_follows_paper(self):
        self.assertTrue(in_severity(0.65, "H"))
        self.assertFalse(in_severity(0.65000001, "H"))
        self.assertFalse(in_severity(0.25, "L"))
        self.assertFalse(in_severity(0.45, "M"))
        self.assertTrue(in_severity(0.25, "M"))
        self.assertTrue(in_severity(0.45, "H"))

    def test_actual_adapter_accepts_exact_65_percent(self):
        source = (REPO / "generation_code/build_occlusion_only.py").read_text()
        tree = ast.parse(source)
        condition = next(node.test for node in ast.walk(tree)
                         if isinstance(node, ast.If) and "bounds[0] <= ratio" in (ast.get_source_segment(source, node.test) or ""))
        expression = compile(ast.Expression(body=condition), "active_H_condition", "eval")
        scope = {"bounds": (.45, .65), "lev": "H", "ratio": .65, "target_area": 1000}
        self.assertTrue(eval(expression, scope))
        scope["ratio"] = .65000001
        self.assertFalse(eval(expression, scope))

    def test_historical_hashes_and_documented_adapter_change(self):
        result = verify_bundle()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["triplet_ast_equal_after_documented_H_endpoint_change"])

    def test_actual_constructor_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, source = fixture(temp)
            generate = load_triplet(source)
            first = generate((1, 101, 1), source.candidates([3]), "dev")
            second = generate((1, 101, 1), source.candidates([3]), "dev")
            ratios = []
            identities = []
            for level in LEVELS:
                self.assertTrue(np.array_equal(first[level][0], second[level][0]))
                self.assertEqual(first[level][1:], second[level][1:])
                row = first[level][2]
                self.assertTrue(in_severity(row["added_occlusion_ratio"], level))
                ratios.append(row["added_occlusion_ratio"])
                identities.append((row["donor_image_id"], row["donor_ann_id"], row["scale"], row["y"]))
            self.assertLess(ratios[0], ratios[1])
            self.assertLess(ratios[1], ratios[2])
            self.assertEqual(len(set(identities)), 1)

    def test_full_annotation_and_pixel_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root, ann, images, _ = fixture(temp)
            result = audit(root, ann, images)
            self.assertEqual(result["generated_images_checked"], 6)
            self.assertEqual(result["pixel_compositions_checked"], 6)
            self.assertEqual(result["original_instances_checked"], 12)

    def test_relocated_archive_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _, _ = fixture(temp)
            relocated = root.parent/"relocated"
            shutil.move(root, relocated)
            self.assertEqual(audit(relocated)["status"], "passed")
            out = render(relocated, "report", None, relocated/"preview.png", crop=True)
            self.assertTrue(out.is_file())

    def test_audit_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _, _ = fixture(temp)
            path = root/"manifests/val_occlusion_report_H.json"
            rows = json.loads(path.read_text())
            rows[0]["donor_ann_id"] += 1
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "identity"):
                audit(root)

    def test_audit_rejects_background_change_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _, _ = fixture(temp)
            path = root/"manifests/val_occlusion_report_L.json"
            rows = json.loads(path.read_text())
            image = Path(rows[0]["generated_file"])
            pixels = cv2.imread(str(image))
            pixels[0, 0] = 0
            cv2.imwrite(str(image), pixels)
            rows[0]["image_sha256"] = sha256(image)
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "outside the donor"):
                audit(root)

    def test_boundary_uses_full_image_seven_pixel_internal_band(self):
        mask = np.ones((720, 1280), bool)
        band = boundary_band(mask)
        self.assertEqual(int(band.sum()), 720*1280-(720-14)*(1280-14))
        self.assertTrue(band[:7].all())
        self.assertFalse(band[7:-7, 7:-7].any())

    def test_boundary_retains_disconnected_components(self):
        mask = np.zeros((20, 20), bool)
        mask[0, 0] = True
        mask[10, 10] = True
        self.assertTrue(np.array_equal(boundary_band(mask), mask))

    def test_matching_is_repeated_at_each_threshold(self):
        preds = [{"category_id": 1}, {"category_id": 1}]
        gt = [{"id": 1, "category_id": 1}, {"id": 2, "category_id": 1}]
        overlaps = np.array([[0.1, 0.6], [0.8, 0.9]])
        self.assertIn(1, associate(preds, gt, overlaps, 0.50))
        self.assertNotIn(1, associate(preds, gt, overlaps, 0.75))

    def test_iou_ties_use_smaller_annotation_id(self):
        gt = [{"id": 7, "category_id": 1}, {"id": 2, "category_id": 1}]
        result = associate([{"category_id": 1}], gt, np.array([[0.8, 0.8]]), 0.5)
        self.assertEqual(result, {2: 0})

    def test_recall_has_no_box_gate_and_quality_unmatched_is_zero(self):
        mask = np.zeros((20, 20), bool)
        mask[2:8, 2:8] = True
        ann = {"id": 1, "category_id": 1, "segmentation": encode(mask)}
        prediction = {"category_id": 1, "score": 0.9, "segmentation": encode(mask), "bbox": [12, 12, 6, 6]}
        score = score_image({"id": 1, "height": 20, "width": 20}, [ann], [prediction], 1)
        self.assertTrue(all(score["recovered"].values()))
        self.assertEqual(score["mask_iou"], 0)
        self.assertEqual(score["boundary_iou"], 0)

    def test_quality_uses_one_box_association_without_mask_filter(self):
        mask = np.zeros((20, 20), bool)
        mask[2:8, 2:8] = True
        ann = {"id": 1, "category_id": 1, "segmentation": encode(mask)}
        predictions = [{"category_id": 1, "score": .9, "segmentation": encode(np.zeros_like(mask)), "bbox": [2, 2, 6, 6]},
                       {"category_id": 1, "score": .8, "segmentation": encode(mask), "bbox": [12, 12, 6, 6]}]
        score = score_image({"id": 1, "height": 20, "width": 20}, [ann], predictions, 1)
        self.assertEqual(score["quality_prediction_rank"], 0)
        self.assertEqual(score["mask_iou"], 0)
        self.assertTrue(all(score["recovered"].values()))

    def test_donor_participates_in_matching(self):
        mask = np.zeros((20, 20), bool)
        mask[2:8, 2:8] = True
        gt = [{"id": 1, "category_id": 1, "segmentation": encode(mask)},
              {"id": 2, "category_id": 1, "segmentation": encode(mask)}]
        pred = {"category_id": 1, "score": .9, "segmentation": encode(mask), "bbox": tight_box(mask)}
        result = score_image({"id": 1, "height": 20, "width": 20}, gt, [pred], 2)
        self.assertFalse(any(result["recovered"].values()))
        self.assertEqual(result["mask_iou"], 0)

    def test_equal_score_preserves_output_order(self):
        mask = np.zeros((20, 20), bool)
        mask[2:8, 2:8] = True
        ann = {"id": 1, "category_id": 1, "segmentation": encode(mask)}
        bad = {"category_id": 1, "score": .9, "segmentation": encode(np.zeros_like(mask)), "bbox": tight_box(mask)}
        good = {"category_id": 1, "score": .9, "segmentation": encode(mask), "bbox": tight_box(mask)}
        first = score_image({"id": 1, "height": 20, "width": 20}, [ann], [bad, good], 1)
        second = score_image({"id": 1, "height": 20, "width": 20}, [ann], [good, bad], 1)
        self.assertEqual(first["mask_iou"], 0)
        self.assertEqual(second["mask_iou"], 1)

    def test_confidence_floor_and_output_limit(self):
        mask = np.ones((20, 20), bool)
        ann = {"id": 1, "category_id": 1, "segmentation": encode(mask)}
        pred = {"category_id": 1, "score": .2999, "segmentation": encode(mask), "bbox": tight_box(mask)}
        score = score_image({"id": 1, "height": 20, "width": 20}, [ann], [pred], 1)
        self.assertFalse(any(score["recovered"].values()))
        with self.assertRaisesRegex(ValueError, "100-instance"):
            score_image({"id": 1, "height": 20, "width": 20}, [ann], [pred]*101, 1)

    def test_quality_mean_includes_unmatched_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _, _ = fixture(temp)
            report = json.loads((root/"annotations/val_occlusion_report_M.json").read_text())
            dev = json.loads((root/"annotations/val_occlusion_dev_M.json").read_text())
            dev_rows = manifest(root, "dev", "M")
            for ann in dev["annotations"]:
                ann["id"] += 1000
            dev_rows[0]["generated_target_ann_id"] += 1000
            predictions = [{"image_id": ann["image_id"], "category_id": ann["category_id"],
                            "segmentation": ann["segmentation"], "bbox": ann["bbox"], "score": .9}
                           for ann in report["annotations"]]
            report["images"] += dev["images"]
            report["annotations"] += dev["annotations"]
            write_json(root/"annotations/val_occlusion_report_M.json", report)
            write_json(root/"manifests/val_occlusion_report_M.json", manifest(root, "report", "M")+dev_rows)
            path = Path(temp)/"predictions.json"
            write_json(path, predictions)
            result = evaluate(root, "report", "M", path)
            self.assertEqual(result["targets"], 2)
            self.assertTrue(all(abs(value-50)<1e-10 for value in result["metrics"].values()))

    def test_prediction_json_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _, _ = fixture(temp)
            data = json.loads((root/"annotations/val_occlusion_report_M.json").read_text())
            predictions = [{"image_id": ann["image_id"], "category_id": ann["category_id"],
                            "segmentation": ann["segmentation"], "bbox": ann["bbox"], "score": .9}
                           for ann in data["annotations"]]
            path = Path(temp)/"predictions.json"
            write_json(path, predictions)
            result = evaluate(root, "report", "M", path)
            self.assertEqual(result["targets"], 1)
            self.assertTrue(all(abs(v-100)<1e-10 for v in result["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
