# M3 synthetic benchmark

Color-only diagnostic; no FashionCLIP/M4 integration or independent user-query improvement claim. Photo suite uses transformed same-image queries and placement masks; synthetic suite uses known fixture masks.

Catalog=8, queries=24, top-k=5, seed=20261107.

| Variant | Hit@1 | Hit@k | Recall@k | MRR | Catalog/query coverage | p50/p95 extraction ms |
|---|---:|---:|---:|---:|---|---|
| rgb_roi | 0.7500 | 0.9167 | 0.9167 | 0.8368 | 1.000/1.000 | 16.24/25.48 |
| hs_roi | 0.3750 | 0.9583 | 0.9583 | 0.5840 | 1.000/1.000 | 0.60/0.78 |
| hsa_roi | 0.5417 | 1.0000 | 1.0000 | 0.7569 | 1.000/1.000 | 0.57/0.76 |
| rgb_mask | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 1.29/2.48 |
| hs_mask | 0.5000 | 1.0000 | 1.0000 | 0.6979 | 1.000/1.000 | 0.57/0.87 |
| hsa_mask | 0.7500 | 1.0000 | 1.0000 | 0.8750 | 1.000/1.000 | 0.53/0.68 |
| hsa_mask_intersection | 0.7500 | 1.0000 | 1.0000 | 0.8750 | 1.000/1.000 | 0.68/1.03 |

Paired cluster bootstrap MRR differences (95% CI; diagnostic only):

- hs_roi -> hsa_roi: 0.1729, CI [0.01666666666666667, 0.37013888888888885]
- hs_mask -> hsa_mask: 0.1771, CI [0.0, 0.38567708333333195]
- hs_roi -> hs_mask: 0.1139, CI [0.03333333333333333, 0.2]
- hsa_roi -> hsa_mask: 0.1181, CI [0.06944444444444445, 0.16666666666666666]
- rgb_roi -> hsa_roi: -0.0799, CI [-0.3611111111111111, 0.21527777777777782]
