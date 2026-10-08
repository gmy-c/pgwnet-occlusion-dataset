# Target evaluation

`tools/evaluate_targets.py` is a repository implementation of manuscript Section 4.4. It is separate from the supplied historical `evaluate_branch.py`, which runs whole-scene COCO AP through a model project. The target evaluator has been tested on synthetic cases; manuscript Table 7 has not been rerun in this update.

## Prediction contract

Run inference on each full 1280 × 720 view using mask threshold 0.50, NMS IoU 0.60, pre-NMS limit 1,000, score floor 0.05 and at most 100 output instances. Export the postprocessed predictions in original output order as a JSON list.

Each item requires:

| Field | Format |
| :--- | :--- |
| `image_id` | Source image ID from the corresponding COCO view |
| `category_id` | Original category ID |
| `score` | Finite confidence in `[0, 1]` |
| `bbox` | Model-predicted COCO `[x, y, width, height]` box |
| `segmentation` | Binary, full-resolution COCO mask RLE after mask thresholding |

Export model boxes together with masks. Some segmentation-only exports omit boxes; join the aligned box and mask outputs before evaluation. Do not replace model boxes with ground-truth boxes or silently derive them from predicted masks. The evaluator does not run a model, resize mask logits, apply NMS or reproduce model-specific preprocessing.

```bash
python tools/evaluate_targets.py \
  --root data/occlusion --split report --level H \
  --predictions outputs/predictions_H.json \
  --out outputs/metrics_H.json
```

Repeat separately for clean/L/M/H. The JSON output records aggregate scores and per-target recovery, mask quality, boundary quality and quality-association rank.

## Matching and denominators

Retain confidence scores at least 0.30, using a stable descending sort. Equal prediction scores preserve output order. Equal ground-truth IoUs select the smaller annotation ID. All scene instances, including donors and non-target objects, participate in matching.

For each mask-IoU threshold from 0.50 to 0.95 in steps of 0.05, start a new one-to-one same-class association pass. Each prediction selects the highest-IoU unmatched ground-truth mask that meets the threshold. There is no box gate. Report R50, R75 and the average of the ten recalls, R50:95.

For quality, start a separate one-to-one same-class association pass using maximum box IoU, requiring 0.50. Ground-truth boxes are tight boxes recomputed from current visible masks. Use the same associated prediction mask for Mask IoU and Boundary IoU, without another mask-IoU filter. An unmatched target receives zero for both scores.

Means divide by all designated targets, not only recovered targets. They do not average categories or pool mask pixels. The report denominator is 300 per view. Multiply all scores by 100 for reporting.

## Boundary implementation

Use the internal band `M minus erode(M, d)`, with

```text
d = max(1, round(0.005 * sqrt(height**2 + width**2)))
```

At 1280 × 720, d is 7. Apply a 3 × 3 erosion kernel seven times with zero padding. Retain every connected component and compute boundary intersection over union at full resolution before any presentation crop.

## Interpretation

R50:95 is a fixed-confidence target recall average, rather than Mask AP or official COCO AR. The quality means combine association success with visible-region accuracy. These target summaries complement whole-scene metrics and do not fully account for background false positives.
