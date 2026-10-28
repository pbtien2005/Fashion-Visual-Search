# Fashion Visual Search Benchmark Suite

This benchmark is the strongest project-level evaluation harness in the repo.
It runs several datasets and produces an aggregate quality report.

## Benchmark Suites

| Suite | Weight | Labels | Purpose |
|---|---:|---|---|
| `auto_smoke` | 0.20 | `evaluation/labels.auto_smoke.json` | Fast API/pipeline smoke test |
| `starter_catalog` | 0.30 | `evaluation/labels.starter_catalog.json` | Regression test with exact product ids from catalog images |
| `real_outfits` | 0.50 | `evaluation/labels.real.json` | Harder user-like outfit benchmark |

`real_outfits` has the highest weight because it is closest to the actual product experience.

## Metrics

| Metric | Meaning |
|---|---|
| `detection_success_rate` | Query images where YOLO detects at least one clothing item |
| `avg_detected_items` | Average number of detected items per query |
| `category_at_5` | Expected category appears in top-5 returned products |
| `color_at_5` | Expected color appears in top-5 returned products |
| `recall_at_5` | Expected exact product id appears in top-5, when labels include exact ids |
| `mrr` | Mean reciprocal rank of the first exact product id hit |
| `avg_latency_ms` | Average upload API latency |
| `failed_queries` | Requests that failed at API/eval level |

For real outfit images, `recall_at_5` and `mrr` can be `n/a` because there is usually no exact product id in the catalog.

## Aggregate Score

Each suite receives a score from available metrics:

```text
suite_score =
  0.20 * detection_success_rate
+ 0.25 * category_at_5
+ 0.20 * color_at_5
+ 0.15 * recall_at_5
+ 0.10 * mrr
+ 0.10 * latency_score
```

Missing metrics, such as real-outfit `recall_at_5`, are skipped and the active weights are normalized.

Latency is converted to a score:

```text
latency_score = 1.0 when avg_latency_ms <= 1000
latency_score = 0.0 when avg_latency_ms >= 8000
```

The overall score is a weighted average of suite scores:

```text
overall_score =
  0.20 * auto_smoke_score
+ 0.30 * starter_catalog_score
+ 0.50 * real_outfits_score
```

## Quality Gates

The benchmark also has pass/fail gates. Gates are intentionally stricter than the score because they catch regressions in specific quality areas.

| Suite | Gate |
|---|---|
| `auto_smoke` | detection >= 0.90, category@5 >= 0.70, color@5 >= 0.70, failed_queries = 0 |
| `starter_catalog` | detection >= 0.95, category@5 >= 0.90, color@5 >= 0.90, recall@5 >= 0.85, MRR >= 0.75, failed_queries = 0 |
| `real_outfits` | detection >= 0.85, category@5 >= 0.70, color@5 >= 0.45, failed_queries = 0 |

If a gate fails, inspect the per-query CSV/JSON before changing the model or reranker.

## Run Full Benchmark

Start services first:

```powershell
docker compose up --build
```

Then run:

```powershell
python evaluation\run_benchmark.py --api-url http://localhost:8000 --run-prefix benchmark_current --timeout 300
```

Outputs:

```text
evaluation/results/benchmark_current_auto_smoke.json
evaluation/results/benchmark_current_starter_catalog.json
evaluation/results/benchmark_current_real_outfits.json
evaluation/results/benchmark_current_summary.json
evaluation/results/benchmark_current_summary.csv
evaluation/results/benchmark_current_summary.md
evaluation/results/latest_benchmark.json
evaluation/results/latest_benchmark.csv
evaluation/results/latest_benchmark.md
```

Run one suite only:

```powershell
python evaluation\run_benchmark.py --api-url http://localhost:8000 --run-prefix quick --suites real_outfits --timeout 300
```

## Latest Run

Latest full run:

```text
run_prefix: benchmark_current
overall_score: 0.8152
overall_gates: FAIL
```

Suite results:

| Suite | Score | Gates | Detection | Category@5 | Color@5 | Recall@5 | MRR | Latency ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| auto_smoke | 0.7377 | FAIL | 1.0000 | 0.6000 | 0.8000 | 0.8000 | 0.8000 | 6059.9 |
| starter_catalog | 0.8992 | FAIL | 1.0000 | 1.0000 | 0.5357 | 1.0000 | 0.9554 | 1244.9 |
| real_outfits | 0.7957 | PASS | 0.9375 | 0.8750 | 0.5000 | n/a | n/a | 1660.9 |

Gate failures:

- `auto_smoke`: `category_at_5=0.6000`, expected `>= 0.70`.
- `starter_catalog`: `color_at_5=0.5357`, expected `>= 0.90`.

Interpretation:

- API stability is good: all suites have `failed_queries = 0`.
- Real outfit suite passes current gates and is the most important user-like benchmark.
- Color matching is the main regression risk, especially on catalog exact-color labels.
- The benchmark is doing its job: it shows where ranking quality still needs work instead of only producing a flattering score.

## Slide Summary

Use this concise version in presentation:

```text
Benchmark = auto_smoke + starter_catalog + real_outfits

Metrics:
- detection_success_rate
- category_at_5
- color_at_5
- recall_at_5 / MRR
- avg_latency_ms

Latest benchmark:
- overall_score: 0.8152
- real_outfits gates: PASS
- failed_queries: 0
- main weakness: color_at_5 on catalog/exact color labels
```
