# Quick start

Run commands from the repository root. Python 3.10 or newer is supported by the active tools. The recorded local test used Python 3.12; see [Testing](TESTING.md).

## Install

```bash
git clone https://github.com/gmy-c/pgwnet-occlusion-dataset.git
cd pgwnet-occlusion-dataset
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_provenance.py
```

## Use the supplied data

Download [PGWNet_Occlusion_Dataset.zip](https://pan.baidu.com/s/16LGOrJ3Cfab2XgAQDZbw0A?pwd=qpst), using extraction code `qpst`. Locate the directory containing `annotations/`, `manifests/`, `images/` and `partitions.json` after extraction; use that directory for `--root`.

The tools consume the actual constructor format described in [Data format](DATA_FORMAT.md). The sharing archive has not been inspected during this update. If it has a different layout, adapt the extraction location before running the commands; do not renumber its IDs or substitute a synthetic example manifest.

```bash
python tools/audit_dataset.py \
  --root data/occlusion \
  --require-paper-counts \
  --out outputs/archive-audit.json

python tools/visualize_dataset.py \
  --root data/occlusion \
  --split report \
  --crop \
  --out outputs/paired-example.png
```

Omit `--crop` for the full-image strip. Add `--image-id ID` to select a specific source image. Cropping is a visualization option only.

## Rebuild from the original source data

Obtain the original BDD8 validation annotation file and every referenced validation image. The frozen input has 998 image records. Keep all source IDs and annotation ordering. Target views from the shared archive do not replace the source pool.

```bash
python tools/build_dataset.py \
  --ann /path/to/ins_seg_val_coco_bdd8_exists.json \
  --images /path/to/BDD100K/10k/val \
  --out data/rebuilt-occlusion \
  --verify-source-hash \
  --audit
```

`data/rebuilt-occlusion` must not exist. The builder writes only clean/L/M/H views for the 60-target development and 300-target report panels. It does not construct appearance, compound or training datasets.

For an existing archive, add source inputs to the audit to check donor-pixel replay:

```bash
python tools/audit_dataset.py \
  --root data/occlusion \
  --source-ann /path/to/ins_seg_val_coco_bdd8_exists.json \
  --source-images /path/to/BDD100K/10k/val \
  --require-paper-counts \
  --out outputs/source-replay-audit.json
```

## Evaluate a checkpoint's exported predictions

Export full-image, postprocessed predictions with `image_id`, `category_id`, `score`, COCO `bbox` and binary mask RLE. Use the manuscript's inference settings before export. See [Evaluation](EVALUATION.md) for the exact contract.

```bash
python tools/evaluate_targets.py \
  --root data/occlusion \
  --split report \
  --level H \
  --predictions outputs/predictions_H.json \
  --out outputs/metrics_H.json
```

Run separately for `clean`, `L`, `M` and `H`, using predictions exported for the corresponding view. Empty prediction lists are valid and give zero target recovery and quality. Model weights and the model inference project are not included in this dataset repository.
