# Validation record

## Executed checks

- All active utilities and supplied Python sources pass syntax compilation.
- Seven historical runtime hashes match both author-provided hash records.
- The legacy helper hash matches the supplied provenance record.
- The active triplet function matches the historical function after the documented paper-standard H endpoint change.
- Synthetic tests exercise the actual active constructor function, COCO writing, source-pixel replay, visible-mask updates, relocated archives, visualization and target metrics.

Run the checks:

```bash
python tools/verify_provenance.py
python -m unittest discover -s tests -v
python -m compileall -q tools generation_code
```

The tests cover deterministic shared donors, annotation and pixel replay, modified identity and background rejection, severity endpoints, stable score ties, smaller-ID IoU ties, donor participation, independent threshold matching, no recall box gate, a separate quality association without a mask filter, zero unmatched scores, confidence filtering and output limits. Boundary tests check the seven-pixel internal band at 1280 × 720, image-edge padding and disconnected components.

The local validation passes **19 tests**. GitHub Actions runs the same checks on Python 3.10 and 3.12.

## Recorded local environment

| Dependency | Version |
| :--- | :--- |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| OpenCV | 5.0.0 (`opencv-python-headless` distribution 5.0.0.93) |
| pycocotools | 2.0.11 |
| Pillow | 12.3.0 |

Install `requirements-tested.txt` in Python 3.12 to reproduce this utility-test environment. This is the environment used for repository validation, not a claim about the historical experiment's environment. The historical bundle's requirements were unpinned. The build wrapper records the actual environment for each new run.

The fixtures are synthetic arrays stored in temporary directories and are never presented as dataset samples. Full reconstruction and model-score reproduction require the original inputs and checkpoints and have not been performed here.
