# Fashion Visual Search Evaluation

This folder contains a small evaluation harness for the upload visual-search API.
It does not depend on the frontend.

For the strongest combined benchmark, see `evaluation/BENCHMARK.md` and run:

```powershell
python evaluation/run_benchmark.py --api-url http://localhost:8000 --run-prefix benchmark_current --timeout 300
```

## Add Query Images

Put fixed query images in:

```text
evaluation/queries/
```

Example:

```text
evaluation/queries/q001.jpg
evaluation/queries/q002.jpg
```

For a real visual-search benchmark, use outfit photos taken from user-like images, not catalog/product images:

```text
evaluation/queries/real_outfits/real_001.jpg
evaluation/queries/real_outfits/real_002.jpg
```

`queries/auto_smoke/` is only a smoke test because those images come from the catalog and are easier than real outfit photos.

You can also build a reproducible starter set from the current catalog/model photos:

```powershell
python evaluation/build_starter_catalog_eval.py --api-url http://localhost:8000 --per-group 4
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.starter_catalog.json --run-name starter_catalog_current
```

This starter set is useful for regression checks, but it is still easier than real outfit photos because each query comes from the product catalog.

## Edit Labels

Use `labels.example.json` as the template and edit `labels.json`.
For the real outfit benchmark, copy `labels.real.example.json` and fill product ids after inspecting the returned results:

```powershell
Copy-Item evaluation/labels.real.example.json evaluation/labels.real.json
```

```json
[
  {
    "query_id": "q001",
    "image": "queries/q001.jpg",
    "expected_items": [
      {
        "type": "top",
        "expected_categories": ["shirt", "top", "blouse"],
        "expected_color": "white",
        "acceptable_product_ids": ["B0EXAMPLE1"]
      }
    ]
  }
]
```

Fields:

- `query_id`: stable id for the query image.
- `image`: path relative to `evaluation/`.
- `expected_items`: garments expected in the image.
- `expected_categories`: category/title terms considered correct for category@5.
- `expected_color`: optional dominant color name, such as `black`, `white`, `blue`, `beige`.
- `acceptable_product_ids`: optional product ids considered correct for Recall@5 and MRR.

If `acceptable_product_ids` is empty or omitted, Recall@5 and MRR are skipped for that expected item.
If `expected_color` is missing, color@5 is skipped for that expected item.

## Run Evaluation

Start the API first, then run:

```powershell
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.json --run-name current
```

Run the real outfit benchmark:

```powershell
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.real.json --run-name real_outfits_current
```

You can build a small real-outfit starter set from Wikimedia Commons:

```powershell
python evaluation/build_wikimedia_real_eval.py --per-category 2
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.real.json --run-name real_outfits_current --timeout 300
```

The builder writes source attribution to `evaluation/real_outfit_sources.md` and `evaluation/real_outfit_sources.json`.

## Tune Rerank Weights

After running an eval, test several image/category/color weight mixes against the returned score breakdown:

```powershell
python evaluation/tune_rerank_weights.py --result evaluation/results/real_outfits_wikimedia_current.json --labels evaluation/labels.real.json
```

This is an offline rerank of products already returned in the eval response, so it is a fast tuning aid rather than a replacement for a full API rerun.

Outputs:

```text
evaluation/results/current.json
evaluation/results/current.csv
evaluation/results/latest_eval.json
evaluation/results/latest_eval.csv
```

## Compare Runs

```powershell
python evaluation/compare_runs.py evaluation/results/v1.json evaluation/results/current.json
```

## Compare Embedding Models

Run the current default FashionCLIP model:

```powershell
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.json --run-name fashionclip_v2
```

To compare against the old CLIP baseline, re-vectorize and reload the catalog with CLIP before restarting the API with the same env:

```powershell
$env:EMBEDDING_MODEL="openai/clip-vit-base-patch32"
$env:EMBEDDING_VERSION="1"
python AI/vectorizer.py
python db/database.py
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.json --run-name clip_v1
```

Compare:

```powershell
python evaluation/compare_runs.py evaluation/results/clip_v1.json evaluation/results/fashionclip_v2.json
```

Do not claim FashionCLIP is better unless the eval metrics improve on your labeled query set.

## Metrics

- `detection_success_rate`: fraction of query images with at least one detected item.
- `avg_detected_items`: average detected items per query.
- `category_at_5`: fraction of labeled expected items whose expected category appears in any top-5 product.
- `color_at_5`: fraction of labeled expected items whose expected color appears in any top-5 product.
- `recall_at_5`: fraction of labeled expected items where an acceptable product id appears in top-5.
- `mrr`: mean reciprocal rank of the first acceptable product id.
- `avg_latency_ms`: average API latency per query.

## Limitations

This harness uses hand-written labels and simple matching against product metadata. It is intended for MVP comparison between versions, not as a statistically strong benchmark.
