# Dataset card

| Property | Description |
| :--- | :--- |
| Name | PGWNet Occlusion Dataset |
| Task | Traffic instance segmentation under controlled added foreground coverage |
| Source | BDD100K validation visible-instance annotations |
| Report panel | 300 source–target pairs, four views per target |
| Development panel | 60 targets, separate frozen target and donor pools |
| Categories | Pedestrian, rider, car, truck, bus, train, motorcycle, bicycle |
| Image resolution | 1280 × 720 |
| Views | Clean, L `[0.10, 0.25)`, M `[0.25, 0.45)`, H `[0.45, 0.65]` |
| Ground truth | Remaining visible masks after foreground donor subtraction |
| Manual annotation | No new manual annotations reported by the historical build receipt |
| Default construction | Frozen deterministic donor search and paired horizontal trajectory |
| Target summaries | R50, R75, R50:95, Mask mIoU, Boundary mIoU |

The report views are repeated observations of 300 targets. Added occlusion is measured against the original visible mask; the dataset does not provide amodal masks. Clean views may already contain natural occlusion.

The fixed validation panel supports checkpoint comparisons and paired failure inspection. It shares the source validation pool used during development. Category counts are uneven; two train targets limit train-specific conclusions. Synthetic copy-paste visibility changes do not establish broad natural-occlusion generalization or retraining stability.

Source imagery and annotations remain subject to BDD100K's data terms. The sharing archive's contents and hashes have not been independently verified in this update. See [Download](README.md#download), [Construction](docs/CONSTRUCTION.md), [Provenance](docs/PROVENANCE.md) and [Evaluation](docs/EVALUATION.md).
