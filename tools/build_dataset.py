"""Preflight the frozen inputs and invoke the unmodified occlusion-only adapter."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

import cv2

from dataset_io import load_json, sha256, write_json
from verify_provenance import verify_bundle

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A directory that does not exist")
    parser.add_argument("--verify-source-hash", action="store_true", help="Require the original COCO JSON byte hash")
    parser.add_argument("--audit", action="store_true", help="Audit all generated labels and pixel composites after building")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output already exists; choose a new directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance = verify_bundle()
    annotation = load_json(args.ann)
    ids = {image["id"] for image in annotation["images"]}
    frozen = load_json(REPO / "generation_code/partitions.json")["legacy_source_partitions"]
    frozen_ids = set().union(*(set(pool) for pool in frozen.values()))
    if len(annotation["images"]) != 998 or len(ids) != 998 or ids != frozen_ids:
        raise ValueError("Input image IDs must match the frozen 998-image validation subset")
    digest = sha256(args.ann)
    expected = load_json(REPO / "generation_code/ORIGINAL_BUILD_COMPLETE.json")["source_annotation_sha256"]["val"]
    if args.verify_source_hash and digest != expected:
        raise ValueError(f"Source annotation SHA-256 mismatch: expected {expected}; got {digest}")
    source_files = []
    for image in annotation["images"]:
        path = args.images / Path(image["file_name"]).name
        pixels = cv2.imread(str(path))
        if pixels is None or pixels.shape[:2] != (image["height"], image["width"]):
            raise ValueError(f"Missing image or incorrect dimensions: {path}")
        if pixels.shape[:2] != (720, 1280):
            raise ValueError(f"Paper reproduction requires 1280 x 720 inputs: {path}")
        source_files.append({"image_id": image["id"], "file_name": path.name, "sha256": sha256(path)})
    environment = {"python": platform.python_version()}
    for package in ["numpy", "opencv-python-headless", "opencv-python", "pycocotools", "Pillow"]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    command = [sys.executable, str(REPO / "generation_code/build_occlusion_only.py"),
               "--ann", str(args.ann.resolve()), "--images", str(args.images.resolve()),
               "--out", str(args.out.resolve())]
    subprocess.run(command, check=True)
    write_json(args.out / "SOURCE_FILES.json", {"annotation_sha256": digest,
               "matches_original_annotation_hash": digest == expected, "images": source_files})
    write_json(args.out / "BUILD_ENVIRONMENT.json", {"environment": environment,
               "source_verification": provenance, "command": command,
               "wrapper_sha256": sha256(Path(__file__)),
               "adapter_sha256": sha256(REPO / "generation_code/build_occlusion_only.py")})
    if args.audit:
        from audit_dataset import audit
        write_json(args.out / "AUDIT.json", audit(args.out, args.ann, args.images, require_paper_counts=True))
    print(f"Created {args.out.resolve()}")


if __name__ == "__main__":
    main()
