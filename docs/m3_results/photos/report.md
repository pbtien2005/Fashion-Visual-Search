# M3 photos benchmark

Color-only diagnostic; no FashionCLIP/M4 integration or independent user-query improvement claim. Photo suite uses transformed same-image queries and placement masks; synthetic suite uses known fixture masks.

Catalog=28, queries=28, top-k=5, seed=20261107.

| Variant | Hit@1 | Hit@k | Recall@k | MRR | Catalog/query coverage | p50/p95 extraction ms |
|---|---:|---:|---:|---:|---|---|
| rgb_roi | 0.7857 | 0.9286 | 0.9286 | 0.8405 | 1.000/1.000 | 47.14/104.44 |
| hs_roi | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 17.16/24.95 |
| hsa_roi | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 9.84/26.86 |
| rgb_mask | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 17.13/31.99 |
| hs_mask | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 13.48/23.06 |
| hsa_mask | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 10.67/16.70 |
| hsa_mask_intersection | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000/1.000 | 8.99/16.01 |

Paired cluster bootstrap MRR differences (95% CI; diagnostic only):

- hs_roi -> hsa_roi: 0.0000, CI [0.0, 0.0]
- hs_mask -> hsa_mask: 0.0000, CI [0.0, 0.0]
- hs_roi -> hs_mask: 0.0000, CI [0.0, 0.0]
- hsa_roi -> hsa_mask: 0.0000, CI [0.0, 0.0]
- rgb_roi -> hsa_roi: 0.1595, CI [0.050595238095238096, 0.2767969877344877]
