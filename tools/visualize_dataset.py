"""Render a real clean/L/M/H pair from the released COCO data and manifests."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_io import dataset, decode, image_path, manifest


def render(root, split, image_id, out, crop=False):
    records = {level: manifest(root, split, level) for level in ("L", "M", "H")}
    if image_id is None:
        image_id = records["L"][0]["source_image_id"]
    rows = {level: next(row for row in values if row["source_image_id"] == image_id)
            for level, values in records.items()}
    clean = dataset(root, split, "clean")
    im = next(image for image in clean["images"] if image["id"] == image_id)
    mask = decode(rows["L"]["original_visible_target"], im["height"], im["width"])
    box = None
    if crop:
        yy, xx = np.where(mask)
        margin = max(48, round(0.5*max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)))
        box = (max(0, xx.min()-margin), max(0, yy.min()-margin),
               min(im["width"], xx.max()+margin+1), min(im["height"], yy.max()+margin+1))
    panel_width, panel_height = 480, 310
    canvas = Image.new("RGB", (4*panel_width, panel_height+90), "#f2f5f9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    labels = [("clean", "Clean", 0.0), ("L", "Light", rows["L"]["added_occlusion_ratio"]),
              ("M", "Moderate", rows["M"]["added_occlusion_ratio"]), ("H", "Heavy", rows["H"]["added_occlusion_ratio"])]
    for index, (level, label, ratio) in enumerate(labels):
        recorded = im["file_name"] if level == "clean" else rows[level]["generated_file"]
        with Image.open(image_path(root, split, level, recorded)) as source:
            source = source.convert("RGB")
            if box:
                source = source.crop(box)
            source.thumbnail((panel_width-24, panel_height-65))
            x = index*panel_width
            canvas.paste(source, (x+(panel_width-source.width)//2, 52+(panel_height-65-source.height)//2))
        draw.text((x+16, 12), f"{label}  |  added coverage {ratio:.1%}", fill="#17304f", font=font)
    draw.text((16, panel_height+12), f"{split} source image {image_id}  |  source target annotation {rows['L']['target_ann_id']}", fill="#17304f", font=small)
    draw.text((16, panel_height+43), "Display crop only; construction and inference use full images." if crop else "Full-image views; resized for display only.", fill="#54667c", font=small)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "report"], default="report")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--crop", action="store_true", help="Use the same display crop for all four views")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.root, args.split, args.image_id, args.out, args.crop))


if __name__ == "__main__":
    main()
