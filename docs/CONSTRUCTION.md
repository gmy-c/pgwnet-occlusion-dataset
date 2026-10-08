# Construction protocol

The active implementation follows Section 4.4 of the manuscript. Historical source files remain unchanged for provenance. The active adapter includes the manuscript's 65% upper endpoint for H.

## Frozen inputs and target ordering

`generation_code/partitions.json` supplies source-image ID pools: 201 development candidates, 697 report candidates and 100 donor images. The donor pool is split into 50 development and 50 report images. These frozen image ID pools are disjoint.

`Source.candidates` accepts non-crowd, non-ignored instances with recorded area at least 128 pixels and a bounding box whose width and height are both at least 8 pixels. The constructor also requires at least 128 decoded target pixels.

`anchors` sorts candidates within each category by a SHA-256-derived key, then traverses categories in round-robin order. It selects at most one target per image. The first 60 valid development triplets and first 300 valid report triplets are retained. A category running out of eligible candidates is not filled by an invented class quota.

## Deterministic donor search

The historical `seed_for` function hashes the string formed by joining these fields with `|`:

```text
bdd-lite-v1.0 | 2131015654 | split | target_annotation_id | fixed_donor_trajectory
```

The first 16 hexadecimal digits are converted to an integer modulo 2^32 and passed to NumPy `RandomState`. Here, spaces illustrate fields; the actual joined string contains no added spaces. The split field is `dev` or `report`.

For each target, the constructor tries up to 100 donor attempts. For each attempt:

1. Draw one donor from the corresponding frozen donor pool. Require at least 128 decoded donor pixels.
2. Crop the donor image and mask to the mask's tight raster extent.
3. Set scale to target height divided by donor crop height, multiplied by a draw from `[1.05, 1.8)`, then clip to `[0.15, 3.0]`.
4. Round resized dimensions with a minimum of two pixels. Reject donors taller than half the image height or wider than half the image width.
5. Resize the donor mask with nearest-neighbor interpolation and image pixels with bilinear interpolation.
6. Choose one bottom-aligned vertical position with a target-height-relative offset drawn from `[-0.05, 0.10)`, clipped to fit in the image.
7. Search up to 100 unique integer horizontal positions between the clipped left-entry and target-right limits.

For L, M and H in order, choose the first eligible position to the right of the preceding selection. The same donor, scale and vertical position serve all three levels. Each selected view must retain at least 64 visible target pixels. If all three levels cannot be found, try another donor; if all 100 attempts fail, reject the target candidate.

The search uses measured mask intersection, not a uniform draw of severity ratios. A horizontally advancing donor need not produce nested covered-pixel sets.

## Coverage and label update

For original visible target V and destination donor mask O:

```text
added_occlusion_ratio = area(V intersect O) / area(V)
updated_target = V minus O
```

The original V is the common reference across levels. L is `[0.10, 0.25)`, M is `[0.25, 0.45)`, and H is `[0.45, 0.65]`.

Paste donor pixels only inside the resized donor mask. For every original scene instance, subtract O, remove empty masks, retain category identity, encode surviving pixels as COCO RLE, and recompute raster area and tight COCO bounding box. Add the donor as an instance, using `source_ann_id = -1`. Assign generated annotation IDs and record the designated target's new ID.

Generated images are full-resolution PNGs. Clean images are copied from the original source files. The manifests retain the original mask, pasted mask, donor transformation, realized ratio, per-instance areas, target mapping and generated-image hash.

## Paper endpoint and historical replay

The archived builder used `r < 0.65` for H. The active adapter uses `r <= 0.65`, following the manuscript. The remaining triplet function is AST-equivalent to the historical function after that documented condition change.

A strict historical replay and a paper-standard rebuild can differ when a candidate realizes exactly 65% coverage. Keep this distinction in any checksum comparison. Do not change frozen IDs, seeds or source annotations to force a matching result.

See [Provenance](PROVENANCE.md) and [Testing](TESTING.md) for verification coverage.
