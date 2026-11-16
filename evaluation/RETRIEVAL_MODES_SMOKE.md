# M4 retrieval catalog evaluation

Run date: 2026-11-16. The live Docker API, YOLO service, PostgreSQL/pgvector,
FashionCLIP and HSV cache were used with the 28-product starter catalog.
The full run uses **all 28 images** in `queries/starter_catalog/` via
`labels.starter_catalog.json`, with the same images and `top_k=5` in all modes.
The earlier seven-image smoke subset is retained below for comparison.

| Mode | API failures | Exact ID in top 5 | MRR | Color@5 | Mean latency |
|---|---:|---:|---:|---:|---:|
| histogram_only | 0/28 | 28/28 | 0.982 | 4/28 | 356 ms |
| embedding_only | 0/28 | 28/28 | 0.982 | 0/28 | 703 ms |
| hybrid | 0/28 | 28/28 | 1.000 | 6/28 | 851 ms |

Across 46 detected crops, the >0.6 threshold rejected one crop in
embedding-only and nine in hybrid. Rejected products are not displayed as
low-confidence suggestions. Full-run artifacts:
[histogram JSON](m4_smoke_results/m4_histogram_full_catalog.json),
[histogram CSV](m4_smoke_results/m4_histogram_full_catalog.csv),
[embedding JSON](m4_smoke_results/m4_embedding_full_catalog.json),
[embedding CSV](m4_smoke_results/m4_embedding_full_catalog.csv),
[hybrid JSON](m4_smoke_results/m4_hybrid_full_catalog.json), and
[hybrid CSV](m4_smoke_results/m4_hybrid_full_catalog.csv).

## Earlier seven-image functional smoke run

`labels.retrieval_smoke.json` contains one catalog image per garment group.

| Mode | Candidate source and score | Acceptance rule | API failures | Exact ID in top 5 | MRR | Mean latency |
|---|---|---|---:|---:|---:|---:|
| histogram_only | All 28 compatible DB products; HSV Hellinger similarity | Any candidate | 0/7 | 7/7 | 0.929 | 369 ms |
| embedding_only | pgvector cosine similarity | Each displayed score > 0.6 | 0/7 | 7/7 | 0.929 | 6243 ms |
| hybrid | pgvector candidates; image/category/HSV/text-style rerank | Each displayed final score > 0.6 | 0/7 | 7/7 | 1.000 | 956 ms |

The seven-image embedding-only run includes first-use FashionCLIP model loading.
Neither set of latencies is a controlled mode comparison. Across 13 detected crops,
the threshold rejected one crop in embedding-only and four in hybrid; rejected
products are not displayed as low-confidence suggestions.

Result artifacts: [histogram JSON](m4_smoke_results/m4_histogram_smoke.json),
[histogram CSV](m4_smoke_results/m4_histogram_smoke.csv),
[embedding JSON](m4_smoke_results/m4_embedding_smoke.json),
[embedding CSV](m4_smoke_results/m4_embedding_smoke.csv),
[hybrid JSON](m4_smoke_results/m4_hybrid_smoke.json), and
[hybrid CSV](m4_smoke_results/m4_hybrid_smoke.csv).

Reproduce the full run with `python evaluation/run_eval.py --api-url http://localhost:8000
--labels evaluation/labels.starter_catalog.json --mode MODE --run-name NAME
--timeout 300`, replacing MODE and NAME for each run. The evaluator writes
JSON/CSV into `evaluation/results/`.

This is a functional smoke run, not a quality claim for user photos. Each query
comes from the same small catalog used for retrieval. The existing evaluator
may count a hit across multiple detections in an image, and its historical
color labels do not directly match the HSV display-label scheme. `color_at_5`
checks whether an expected color **name** exactly equals any returned product's
`dominant_color_name`; it does not measure histogram similarity. Consequently
the full-run Color@5 values (4/28, 0/28, 6/28) are not a calibrated color
comparison.

For histogram-only retrieval, every query's exact catalog product appeared in
the top five, with nonzero HSV `color_score` (0.592–0.895). Yet all seven exact
products had a different display color name from their expected label:

| Product ID | Expected color | HSV display color | HSV similarity |
|---|---|---|---:|
| B0GGYQ4J5Z | beige | white | 0.889 |
| B0D2K2FWDZ | brown | red | 0.884 |
| B0C3838ZBF | gray | white | 0.885 |
| B076JFJCL4 | black | white | 0.865 |
| B07HNVV8MV | beige | white | 0.592 |
| B0G6CZ3RV9 | black | white | 0.894 |
| B0FZJ583QN | green | white | 0.895 |

The catalog color cache has 28/28 descriptors with `roi_source=whole_catalog_photo`
and `mask_status=missing`. Thus the HSV display name can reflect the photo's
background, especially white, instead of the garment. The starter expected
color labels also need independent review. To assess garment color accuracy,
create or verify garment masks/ROIs and human color labels, rebuild the cache,
then rerun the color evaluation. Do not interpret the current `color_at_5=0` as
zero histogram similarity or as a reliable garment-color accuracy estimate.
The 0.6 threshold is an initial rule requested for this implementation, not
a value tuned on independent validation data.
