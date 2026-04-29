# Fashion Visual Search Evaluation

This folder contains a small evaluation harness for the upload visual-search API.
It does not depend on the frontend.

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

## Edit Labels

Use `labels.example.json` as the template and edit `labels.json`.

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
