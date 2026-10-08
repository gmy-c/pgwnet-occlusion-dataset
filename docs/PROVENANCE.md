# Provenance

## Author-supplied code

The uploaded construction ZIP contains fourteen files. The repository retains the seven `original_runtime/` scripts and the legacy helper byte-for-byte. Their SHA-256 values match the supplied records. `metadata/source_bundle.json` records the uploaded ZIP hash and the original hashes of all fourteen files.

`ORIGINAL_BUILD_COMPLETE.json` describes a broader historical build, including appearance, compound and training artifacts. It is retained as evidence of source provenance; those additional datasets are not produced by the active dataset-only entry point. `CODE_PROVENANCE.json` describes the supplied bundle before this repository update.

The active `build_occlusion_only.py` was supplied as a new extraction, not an original hash-listed script. This release updates its H upper bound to include 0.65 according to the paper and replaces its header to identify that change. Its source annotations, seed function, candidate ordering, donor search, resizing, pixel copy and instance update logic remain as supplied. `tools/verify_provenance.py` compares its triplet AST with the historical function after the documented endpoint change.

The English bundle README is a translation and update. Newly added wrapper, audit, visualization, target evaluator and tests are repository utilities, not historical experiment files.

## Manuscript and figures

The protocol and reference scores follow the latest author-provided manuscript, Section 4.4 and Table 7. The motorcycle paired strip and animation use Figure 8(b); the six-panel pedestrian construction illustration uses Figure 8(a), a different scene. Distribution charts use reported category and original-visible-size counts. Reference charts use Table 7 values.

No generated scene imagery is used. The figures are display examples; full-resolution images, exact donor transforms and source IDs must come from the dataset and construction manifests.

## Verification boundary

This release verifies source hashes, the documented adapter change and synthetic behavioral tests. It does not claim a full rebuild from the original 998-image input, a byte-for-byte audit of the sharing archive, or a rerun of Table 7. Those checks require the original source images and annotations, and, for model results, the corresponding inference project and weights.
