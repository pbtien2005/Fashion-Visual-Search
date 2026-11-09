# Fashion Visual Search

Fashion Visual Search is an end-to-end AI/full-stack web app for finding similar fashion products from an uploaded outfit image.

Production color is **HSV Histogram 147**, with the existing FashionCLIP encoder. Query/catalog require a matching versioned HSV sidecar; there is no Dominant RGB fallback. See [production migration and deployment](docs/M3_HSV_PRODUCTION.md) and the [M4 review boundary](docs/M4_HSV_INTEGRATION_REVIEW.md). Docker requires the rebuilt cache file before startup.

```text
Upload outfit image
-> YOLO detects clothing items
-> each crop is encoded by the configured CLIP/FashionCLIP model
-> PostgreSQL + pgvector retrieves similar products
-> API reranks by image/category/color/text-style scores and penalties
-> React frontend renders detected items and product cards
```

Detailed submission report: [docs/REPORT.md](docs/REPORT.md)

Demo checklist: [docs/DEMO_CHECKLIST.md](docs/DEMO_CHECKLIST.md)

Evaluation template: [docs/EVALUATION_TEMPLATE.md](docs/EVALUATION_TEMPLATE.md)

## Tech Stack

- Frontend: React 19 + Vite.
- API: FastAPI, httpx, Pydantic.
- Object detection: YOLO through `ultralytics`, OpenCV crop processing.
- Embedding: Hugging Face `transformers` CLIP interface, default `patrickjohncyh/fashion-clip`.
- Vector DB: PostgreSQL 16 + pgvector.
- Storage: Cloudflare R2 / S3-compatible API through `boto3`.
- ETL: Apify Amazon scraper + R2 uploader.
- Evaluation: Python upload API benchmark with JSON/CSV reports.

## Folder Structure

```text
AI/                    Embedding model, vectorizer, color utilities
api/                   Main FastAPI service and search orchestration
db/                    PostgreSQL schema/load/search helpers and migrations
etl/                   Apify scraper, R2 uploader, ETL pipeline
evaluation/            Evaluation harness, labels, query images, results
models/object-detection/ Local YOLO weight location
object_detection/      YOLO detection FastAPI service
web/fe/                React/Vite frontend
docs/                  Report, demo checklist, evaluation template
docker-compose.yml     Runtime service composition
```

## Current Inspected Data

On this machine, the running API returned:

```text
GET http://localhost:8000/api/products?limit=1&offset=0
total = 850
```

Local files inspected:

- `data/fashion_metadata.json`: 1580 metadata records.
- `data/fashion_with_vectors.json`: 850 vectorized records.
- Embedding dimension: 512.
- Embedding model/version in vectorized file: `patrickjohncyh/fashion-clip`, version `2`.

If the Docker volume is reset or an older DB is used, re-check the product count before demo.

## Environment Variables

Create `.env` from the example:

```powershell
copy .env.example .env
```

Important variables:

```env
DB_HOST=postgres-pgvector
DB_PORT=5432
DB_USER=pbtien
DB_PASSWORD=change_me
DB_NAME=fashion_db

OBJECT_DETECTION_URL=http://object-detection:8001
OBJECT_DETECTION_MODEL_PATH=/models/object-detection/best.pt
OBJECT_DETECTION_CONF=0.5

API_PORT=8000
VITE_API_URL=http://localhost:8000
MATCH_SCORE_THRESHOLD=0.70
MIN_IMAGE_SIMILARITY=0.60
ENABLE_TEXT_STYLE_RERANK=true
RERANK_IMAGE_WEIGHT=0.60
RERANK_CATEGORY_WEIGHT=0.15
RERANK_COLOR_WEIGHT=0.15
RERANK_TEXT_STYLE_WEIGHT=0.10

EMBEDDING_MODEL=patrickjohncyh/fashion-clip
EMBEDDING_VERSION=2

APIFY_API_TOKEN=
R2_ENDPOINT_URL=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_PUBLIC_BASE_URL=
```

Use Docker service names for Compose (`DB_HOST=postgres-pgvector`, `DB_PORT=5432`). Use host values when running Python scripts against Docker Postgres from the host:

```env
DB_HOST=localhost
DB_PORT=5433
```

Do not expose R2 credentials to the frontend. Only `VITE_` variables are available in browser code.

## YOLO Model

Place the trained YOLO model here:

```text
models/object-detection/best.pt
```

Docker Compose mounts it into the object detection container at:

```text
/models/object-detection/best.pt
```

If the model is missing, `POST /detect/upload` returns a 503 error.

## Run with Docker

Start all services:

```powershell
docker compose up --build
```

URLs:

- Frontend: http://localhost:5173
- Main API health: http://localhost:8000/health
- Object detection health: http://localhost:8001/health
- PostgreSQL host port: `localhost:5433`

Service containers:

- `fashion_frontend`
- `fashion_api`
- `fashion_object_detection`
- `fashion_postgres_pgvector`

## Health Checks

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod "http://localhost:8000/api/products?limit=1&offset=0"
```

Expected:

- Main API returns `success=true`.
- Object detection returns `model_available=true`.
- Product endpoint returns positive `total` after data is loaded.

## Run ETL

Install Python dependencies first if running scripts locally:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Run scraper + R2 upload:

```powershell
python etl\test.py
```

Useful limits:

```powershell
$env:ETL_MAX_ITEMS_PER_KEYWORD="20"
$env:ETL_KEYWORD_START="0"
$env:ETL_KEYWORD_LIMIT="5"
python etl\test.py
```

Output:

```text
data/fashion_metadata.json
data/scraped_history.json
```

## Vectorize Products

Generate FashionCLIP embeddings and HSV147 descriptors (new output files only):

```powershell
python -m AI.vectorizer
```

Output:

```text
data/fashion_with_hsv_vectors.json
data/color_cache_hsv147.json
```

Default embedding config:

```env
EMBEDDING_MODEL=patrickjohncyh/fashion-clip
EMBEDDING_VERSION=2
```

The database schema expects:

```sql
embedding VECTOR(512)
```

Do not switch to a model with a different output dimension without a deliberate schema/migration plan.

## Load Database

If using Docker Postgres from host scripts, set:

```powershell
$env:DB_HOST="localhost"
$env:DB_PORT="5433"
$env:DB_USER="pbtien"
$env:DB_PASSWORD="change_me"
$env:DB_NAME="fashion_db"
```

Then load:

```powershell
python db\database.py
```

Optional migrations:

```powershell
python db\run_migrations.py
```

Rebuild only color for an existing catalog, without re-embedding or overwriting the original:

```powershell
python -m AI.rebuild_color_cache --catalog data/fashion_with_vectors.json --allow-remote --output data/color_cache_hsv147_run001.json
```

## Run Pieces Locally

Object detection:

```powershell
uvicorn object_detection.api:app --host 0.0.0.0 --port 8001
```

Main API:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd web\fe
npm install
npm run dev
```

## API Endpoints

Main API:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health and object detection reachability |
| POST | `/api/search/upload` | Upload outfit image, detect garments, search products |
| GET | `/api/products` | List products with optional filters |
| GET | `/api/products/{product_id}/image` | Backend image proxy for R2/S3 product images |

Object detection service:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Detection model health |
| POST | `/detect/upload` | Detect clothing and return crop images |

Example upload:

```powershell
curl.exe -X POST "http://localhost:8000/api/search/upload?top_k=5" `
  -F "file=@path\to\outfit.jpg"
```

## Frontend Behavior

The frontend has three tabs:

- Search: upload outfit image and render detected items.
- Products: browse all catalog products and filter by title/brand/category.
- History: save recent searches in browser `localStorage`, capped at 20 entries.

Product images use `usable_image_url`. If the API returns a relative path such as `/api/products/{id}/image`, the frontend resolves it to `http://localhost:8000/api/products/{id}/image`.

## Evaluation

Run the current evaluation harness:

```powershell
python evaluation\run_eval.py --api-url http://localhost:8000 --labels evaluation\labels.json --run-name current
```

Run auto-smoke labels:

```powershell
python evaluation\run_eval.py --api-url http://localhost:8000 --labels evaluation\labels.auto_smoke.json --run-name fashionclip_v2_api_current
```

Compare runs:

```powershell
python evaluation\compare_runs.py evaluation\results\clip_v1_auto_smoke.json evaluation\results\latest_eval.json
```

Stored latest local result:

- `detection_success_rate`: 1.0
- `category_at_5`: 0.8
- `color_at_5`: 1.0
- `recall_at_5`: 0.9
- `mrr`: 0.85
- `avg_latency_ms`: about 373 ms

Treat these as small smoke-set results, not a full benchmark.

## Troubleshooting

### Missing YOLO model

Put `best.pt` at:

```text
models/object-detection/best.pt
```

Then restart the object detection service.

### Product list is empty

Load vectorized data:

```powershell
python AI\vectorizer.py
python db\database.py
```

When running from host against Docker Postgres, use `DB_HOST=localhost` and `DB_PORT=5433`.

### Product images do not render

Stored product images may be `s3://...` paths. Browsers cannot load those directly.

Options:

- Configure backend R2 credentials so `/api/products/{id}/image` can stream images.
- Or configure a public R2/custom domain:

  ```env
  R2_PUBLIC_BASE_URL=https://your-public-r2-domain
  ```

Never put R2 secrets into frontend env vars.

### Embedding model download fails

The first FashionCLIP run downloads from Hugging Face. It may need network access and cache write access.

Check:

```text
%USERPROFILE%\.cache\huggingface
```

### Low-confidence result

If the API returns:

```text
Detected item, but no confident product match found
```

that is expected behavior when top match score is below `MATCH_SCORE_THRESHOLD` or image similarity is below `MIN_IMAGE_SIMILARITY`.

### No clothing detected

The API returns success with an empty `detected_items` array. Use a clearer outfit image or check object detection health.

## Verification Commands

```powershell
python -m compileall AI api db etl object_detection evaluation
cd web\fe
npm run build
```

Health/runtime checks:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod "http://localhost:8000/api/products?limit=1&offset=0"
```

## Known Limitations

- Current inspected catalog has 850 searchable products, which is still small.
- Catalog balance depends on the ETL keywords in `etl/test.py`.
- Outfit photos and Amazon catalog images have a domain gap.
- YOLO bbox crops are rectangular and not segmentation masks.
- Color uses mask-aware HSV147, with explicit ROI/missing-descriptor policies. H-S discards chromatic brightness; display color names are approximate. Real garment masks still depend on M2.
- Reranking uses metadata keyword heuristics.
- Amazon product URLs can become stale or out of stock.
- Evaluation set is small and should be expanded before making strong quality claims.
- Segmentation/SAM is planned future work, not implemented in this repo.

## Submission Docs

- [Detailed report](docs/REPORT.md)
- [Demo checklist](docs/DEMO_CHECKLIST.md)
- [Evaluation template](docs/EVALUATION_TEMPLATE.md)
