# Constructor output format

This describes the actual files written by `generation_code/build_occlusion_only.py`, not a proposed interchange schema. The downloaded sharing archive still needs a layout check against these files.

## Files

| Location | Contents |
| :--- | :--- |
| `annotations/val_occlusion_{split}_{level}.json` | COCO images, full-scene instances and categories |
| `manifests/val_occlusion_{split}_{level}.json` | One construction record per source target, for L/M/H |
| `images/val_occlusion_{split}_{level}/` | Full-resolution generated PNGs or copied clean source files |
| `partitions.json` | Frozen source and donor image ID pools |
| `OCCLUSION_BUILD.json` | Adapter counts and rejected-candidate totals |
| `SOURCE_FILES.json` | Source image hashes and annotation hash, added by the preflight wrapper |
| `BUILD_ENVIRONMENT.json` | Library versions, command and code hashes, added by the wrapper |
| `AUDIT.json` | Post-build audit results, when the wrapper is run with `--audit` |

`split` is `dev` or `report`; `level` is `clean`, `L`, `M` or `H`. There are eight COCO annotation files and six occluded-view manifest files. Clean target identity is obtained from the corresponding L manifest's original target ID.

## COCO annotations

Image IDs are preserved from the source. Category IDs retain the original BDD8 mapping. Generated annotation IDs are assigned by each view's writer; compare them through the source mapping, rather than assuming that IDs have equal meanings across files.

Each generated original-instance annotation contains `source_ann_id`. Its RLE segmentation describes remaining visible pixels, and its area and `[x, y, width, height]` box are recomputed from that mask. The donor uses `source_ann_id = -1`. Empty original instances are removed. Clean annotations retain the original source representation.

Masks occupy the full original image coordinate system. RLE `counts` is an ASCII string and `size` is `[height, width]`.

## Construction records

| Field | Meaning |
| :--- | :--- |
| `protocol` | Historical constructor namespace, `bdd-lite-v1.0` |
| `source_image_id`, `source_file` | Original image identity and recorded path |
| `target_ann_id`, `target_category_id` | Original designated target identity and category |
| `generated_target_ann_id` | Target annotation ID in this generated COCO file |
| `donor_image_id`, `donor_ann_id`, `donor_split` | Donor source identity and pool |
| `scale`, `x`, `y` | Donor resizing factor and destination top-left placement |
| `trajectory` | `fixed_asset_left_entry` |
| `family`, `level` | Foreground occlusion family and L/M/H view |
| `added_occlusion_ratio` | Realized coverage relative to original visible target area |
| `original_visible_target`, `occluder` | Full-resolution RLE masks V and O |
| `changes` | Source annotation ID and before/after raster area for every original instance |
| `generated_file` | Recorded generated-image path |
| `image_sha256` | Generated PNG file SHA-256 |

Some historical records use `sha256` in addition to `image_sha256`; the audit accepts either field.

## Paths after moving a dataset

The author writer records absolute paths. Active tools first resolve each image under the local `--root/images/val_occlusion_{split}_{level}/` directory using its basename, then fall back to the recorded path. This keeps extracted datasets usable without modifying their annotation identities or construction records.

For source-pixel replay, `--source-images` similarly supplies the original source directory by basename. Exact original IDs and mask annotations are still required.
