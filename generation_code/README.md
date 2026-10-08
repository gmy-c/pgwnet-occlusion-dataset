# Dataset construction sources

This directory contains the author's supplied construction bundle. The default entry point builds only the paired foreground-occlusion panel. The historical runtime is retained for provenance.

## Entry point

From the repository root, use the preflight wrapper:

```bash
python tools/build_dataset.py \
  --ann /path/to/ins_seg_val_coco_bdd8_exists.json \
  --images /path/to/BDD100K/10k/val \
  --out data/occlusion \
  --verify-source-hash \
  --audit
```

The output directory must not exist. The wrapper checks the frozen image IDs, original annotation byte hash, image dimensions and source-code provenance, then invokes `build_occlusion_only.py`. It records the environment and source file hashes and optionally audits generated outputs.

The underlying adapter can also be invoked directly:

```bash
python generation_code/build_occlusion_only.py \
  --ann /path/to/ins_seg_val_coco_bdd8_exists.json \
  --images /path/to/BDD100K/10k/val \
  --out data/occlusion
```

The direct adapter omits the wrapper's preflight, environment receipt and post-build audit.

## Source requirements

Replay requires all 998 original validation images referenced by the frozen BDD8 COCO input. Preserve image, annotation and category IDs, filename basenames, mask representations and annotation order. The previously generated target views cannot replace the complete candidate and donor pool.

The expected original validation annotation SHA-256 is:

```text
ea781d49f00429429b769960cc7146d99c97e4e385901f0e78faf19b502b23c2
```

BDD100K's nominal validation split has 1,000 images; the historical COCO input used by this builder has 998 records. Replacing that input with a newly converted or renumbered annotation file does not reproduce the frozen selection.

## Files and provenance

| File | Purpose |
| :--- | :--- |
| `build_occlusion_only.py` | Extracted occlusion-only adapter, updated to include 65% in H according to the manuscript |
| `partitions.json` | Frozen source-image partitions and disjoint donor pools |
| `original_runtime/build_lite.py` | Historical complete builder containing the paired occlusion loop |
| `original_runtime/lite_operators.py` | Historical version, level order and deterministic seed function |
| `legacy/tools/build_protocol.py` | Source decoding, candidate selection, RLE encoding and COCO writing helpers |
| `ORIGINAL_BUILD_COMPLETE.json` | Author-provided receipt for the historical full build |
| `CODE_PROVENANCE.json` | Author-provided source-file hash record |

All seven files in `original_runtime/` remain byte-for-byte identical to the supplied bundle and match both supplied hash records. The legacy helper matches `CODE_PROVENANCE.json`; its historical identity is not covered by the original build's seven-file list. The supplied receipts are provenance records, rather than an independent rerun.

The active adapter changes the historical H condition from `0.45 <= r < 0.65` to **`0.45 <= r <= 0.65`**. All other triplet sampling and compositing logic matches the historical function at the AST level. The provenance checker tests this explicitly. If a candidate placement realizes exactly 65%, the current adapter may choose a different H view from the historical build.

## Historical integrations

The archived `evaluate_branch.py`, `make_branch_config.py`, `lite_dataset.py` and `validate_branch.py` belong to the original, broader MMDetection experiment. They require the separate model project, its framework environment and additional training/appearance/compound artifacts. They are not the default dataset-only workflow and are not a target-metric implementation for manuscript Table 7.

For the active workflow, see [Quick start](../docs/QUICKSTART.md), [Construction](../docs/CONSTRUCTION.md), [Data format](../docs/DATA_FORMAT.md) and [Evaluation](../docs/EVALUATION.md).
