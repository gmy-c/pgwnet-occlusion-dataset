<div align="center">

# PGWNet Occlusion Dataset

### Paired foreground occlusion for traffic instance segmentation

**Same target · Four visibility levels · Updated visible-instance annotations**

![Report targets: 300](https://img.shields.io/badge/Report_targets-300-3569B0?style=flat-square)
![Development targets: 60](https://img.shields.io/badge/Development_targets-60-7864B3?style=flat-square)
![Paired views: 4](https://img.shields.io/badge/Paired_views-4-16836B?style=flat-square)
![Traffic categories: 8](https://img.shields.io/badge/Traffic_categories-8-B07828?style=flat-square)

[**Download dataset**](https://pan.baidu.com/s/16LGOrJ3Cfab2XgAQDZbw0A?pwd=qpst) · [Overview](#overview) · [Construction](#construction) · [Evaluation](#evaluation) · [Citation](#citation)

**Baidu Netdisk extraction code: `qpst`**

</div>

---

## Overview

The PGWNet Occlusion Dataset is a **controlled foreground-occlusion evaluation panel** for traffic instance segmentation, introduced in Section 4.4 of *PGWNet: Progressive Gated Wavelet Fusion and Boundary-Aware Mask Supervision for Traffic Instance Segmentation*.

It follows the **same designated traffic target** across a clean image and three levels of added foreground coverage. The resulting pairs support a direct question: how well does a model recover the target and segment its remaining visible pixels as visibility decreases?

**Models receive the full image.** Evaluation tracks designated targets across views, while all scene instances—including pasted donors—participate in prediction matching.

| Property | Description |
|---|---|
| Source | BDD100K validation images and visible-instance annotations |
| Task | Traffic instance segmentation under controlled added foreground occlusion |
| Report panel | 300 source–target pairs |
| Development panel | 60 targets; target identities and donor pools separated from the report panel |
| Paired views | Clean, light, moderate and heavy |
| Report target-view records | 1,200, derived from 300 targets × 4 views |
| Full-image dimensions | 1280 × 720 |
| Annotation semantics | Remaining **visible** regions; no inferred amodal completion |

The 1,200 count describes target-view records, not a verified count of unique image files or independent source scenes. Target and donor separation does not imply that every source scene is disjoint between panels.

## Download

### Dataset archive

| Item | Access information |
|---|---|
| File | **PGWNet_Occlusion_Dataset.zip** |
| Download | **[Open Baidu Netdisk](https://pan.baidu.com/s/16LGOrJ3Cfab2XgAQDZbw0A?pwd=qpst)** |
| Extraction code | **`qpst`** |

**中文下载提示：** 点击上方百度网盘链接，输入提取码 **qpst**，下载 `PGWNet_Occlusion_Dataset.zip`。

The sharing link is provided by the author. Archive accessibility, contents, checksum and file-layout correspondence have not been independently verified during preparation of this README. The statistics and reference scores below come from the supplied manuscript.

## Paired visibility levels

Let **V** denote the target's original visible mask and **O** the pasted foreground donor mask. Added coverage is measured against the original visible region:

$$
r_{\mathrm{added}} = \frac{|V \cap O|}{|V|},
\qquad G = V \setminus O.
$$

Here, **G** is the updated visible target mask.

| View | Added coverage | Interpretation |
|---|---:|---|
| **Clean** | 0% | Original view, without added synthetic occlusion |
| **Light · L** | [10%, 25%) | Limited added foreground coverage |
| **Moderate · M** | [25%, 45%) | Intermediate added foreground coverage |
| **Heavy · H** | [45%, 65%] | Larger added foreground coverage |

Clean images can already contain natural occlusion. The ratio above measures **additional** coverage of the original visible mask; it does not estimate the hidden fraction of a complete object.

## Construction

1. **Select the target and donor.** Choose a designated traffic instance and a foreground donor from another image.
2. **Build paired views.** Keep target identity, donor identity and donor scale fixed across L/M/H. Translate the donor to reach each coverage interval.
3. **Update visible annotations.** Subtract the pasted donor mask from every affected original instance. Remove empty instances and add the donor as a foreground instance.
4. **Recompute geometry.** Calculate current visible-mask areas and tight bounding boxes from the updated annotations.

The donor's position may change which pixels are covered. Increasing coverage therefore does not require the occluded pixels to be nested between severity levels. The construction preserves visible-region semantics rather than reconstructing hidden object shapes.

## Report-panel composition

### Categories

| Pedestrian | Rider | Car | Truck | Bus | Train | Motorcycle | Bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 17 | 61 | 58 | 56 | 2 | 22 | 20 |

### Target sizes

| Small | Medium | Large | Total |
|---:|---:|---:|---:|
| 136 | 115 | 49 | **300** |

Size groups use COCO thresholds on the **original visible target area**, before added occlusion. Development-panel category counts are not specified in the manuscript.

## Evaluation

### Full-image inference

Inference uses complete **1280 × 720** images. Ground-truth target crops and clipping predictions to ground-truth boxes or masks are outside this protocol.

| Setting | Value |
|---|---:|
| Mask binarization threshold | 0.50 |
| NMS IoU | 0.60 |
| Pre-NMS prediction limit | 1,000 |
| Inference score floor | 0.05 |
| Maximum output instances | 100 |
| Confidence retained for target evaluation | ≥ 0.30 |

### Target metrics

| Metric | Definition |
|---|---|
| **Target R50** | Fraction of designated targets recovered at mask IoU ≥ 0.50 |
| **Target R75** | Fraction recovered at mask IoU ≥ 0.75 |
| **Target R50:95** | Mean target recall over 10 mask-IoU thresholds, 0.50:0.05:0.95 |
| **Target Mask mIoU** | Mean visible-mask IoU over all designated targets, with unmatched targets scored as zero |
| **Target Boundary mIoU** | Mean internal-boundary-band IoU over all designated targets, with unmatched targets scored as zero |

**Recall matching:** at each threshold, independently process predictions in descending confidence order and match to the highest-mask-IoU unmatched same-class instance. Matching is one-to-one, without a preliminary box-IoU gate.

**Quality matching:** run a separate score-ordered, same-class one-to-one association using maximum box IoU ≥ 0.50. Ground-truth boxes come from current visible masks. Use the same associated prediction for Mask IoU and Boundary IoU, without an additional mask-IoU filter.

Both paths include donors and non-target instances during matching. Only designated targets enter the reported summaries. Retain original prediction order for confidence ties; choose the smaller annotation ID for equal-IoU ground-truth candidates.

Boundary IoU uses the internal band `M \ erode(M, d)`, with `d = max(1, round(0.005 × sqrt(H² + W²)))`. At 1280 × 720, **d = 7**: apply seven iterations of 3 × 3 erosion with zero padding and retain every component. Quality averages divide by all 300 report targets, including unmatched ones.

**Target R50:95 is a fixed-confidence recall average, not Mask AP or official COCO AR.**

<details>
<summary><strong>Manuscript reference results — Table 7</strong></summary>

All values use a 0–100 scale. These are reported manuscript scores, rather than results reproduced by this repository.

| View | Method | R50 | R75 | R50:95 | Mask mIoU | Boundary mIoU |
|---|---|---:|---:|---:|---:|---:|
| Clean | RTMDet-R | 58.33 | 41.33 | 38.77 | 47.72 | 36.61 |
| Clean | **PGWNet** | **62.00** | **43.00** | **41.13** | **50.53** | **38.86** |
| Light | RTMDet-R | 51.67 | 35.00 | 32.60 | 41.39 | 30.72 |
| Light | **PGWNet** | **55.00** | **38.33** | **35.57** | **44.62** | **34.28** |
| Moderate | RTMDet-R | 43.33 | 26.67 | 24.80 | 33.77 | 25.11 |
| Moderate | **PGWNet** | **47.33** | **31.33** | **29.40** | **38.11** | **29.28** |
| Heavy | RTMDet-R | 24.67 | 10.33 | 12.83 | 18.87 | 14.68 |
| Heavy | **PGWNet** | **30.33** | **18.33** | **17.03** | **22.29** | **18.03** |

</details>

## Scope and interpretation

This is a controlled **validation** panel derived from a source validation pool used during model development. Its four views are repeated observations of the same targets. The category distribution is uneven, including only two train targets, and target averages do not fully penalize background false positives.

The panel supports paired checkpoint comparisons under added foreground occlusion. It does not establish independent natural-occlusion generalization, amodal reconstruction, or deployment safety. The reported differences describe the evaluated checkpoints; retraining variability requires separate study.

## Data rights

The dataset derives from BDD100K. Use of source images and annotations remains subject to the original data terms and any applicable author-provided release terms. This README does not grant a new license over the underlying imagery.

## Citation

If you use this evaluation resource, acknowledge the companion PGWNet manuscript and its BDD100K source. The PGWNet manuscript is currently a submission draft; no accepted venue, DOI or arXiv identifier is asserted.

```bibtex
@misc{gong_pgwnet_submission,
  author = {Gong, Mingyu and Chen, Gengbiao and Zhang, Jinlai and Li, Yucheng},
  title = {{PGWNet}: Progressive Gated Wavelet Fusion and Boundary-Aware Mask Supervision for Traffic Instance Segmentation},
  year = {2026},
  note = {Unpublished manuscript; controlled occlusion evaluation described in Section 4.4}
}

@inproceedings{yu2020bdd100k,
  author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  title = {{BDD100K}: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```

---

<div align="center">

**Maintainer: Mingyu Gong** · [gongmingyu@stu.csust.edu.cn](mailto:gongmingyu@stu.csust.edu.cn)

[Download dataset](https://pan.baidu.com/s/16LGOrJ3Cfab2XgAQDZbw0A?pwd=qpst) · Extraction code **qpst**

</div>