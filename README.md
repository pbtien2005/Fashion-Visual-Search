# Fashion Visual Search

MVP web app for fashion visual search:

```text
Upload outfit image
-> YOLO object detection detects and crops each clothing item
-> FashionCLIP vectorizes each crop
-> PostgreSQL + pgvector searches similar products
-> React frontend renders results grouped by detected item
```

## Services

- `frontend`: React/Vite upload and results UI, port `5173`.
- `api`: FastAPI upload pipeline, port `8000`.
- `object-detection`: FastAPI YOLO crop service, port `8001`.
- `postgres-pgvector`: PostgreSQL 16 with pgvector, host port `5433`.

## Environment

Create a local env file from the example:

```powershell
copy .env.example .env
```

For Docker Compose, the default `.env.example` values match the service names:

```env
DB_HOST=postgres-pgvector
DB_PORT=5432
DB_USER=pbtien
DB_PASSWORD=change_me
DB_NAME=fashion_db

OBJECT_DETECTION_URL=http://object-detection:8001
OBJECT_DETECTION_MODEL_PATH=/models/object-detection/best.pt

API_PORT=8000
VITE_API_URL=http://localhost:8000
MATCH_SCORE_THRESHOLD=0.45

EMBEDDING_MODEL=patrickjohncyh/fashion-clip
EMBEDDING_VERSION=2

APIFY_API_TOKEN=
R2_ENDPOINT_URL=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_PUBLIC_BASE_URL=
```

Do not put real secrets in frontend env vars. Only variables prefixed with `VITE_` are exposed to the browser.

## YOLO Model

Place the object detection weight file here:

```text
models/object-detection/best.pt
```

The file is intentionally ignored by Git. Docker mounts it into the object detection container at:

```text
/models/object-detection/best.pt
```

If the model is missing, `/detect/upload` returns:

```text
Missing YOLO model at /models/object-detection/best.pt
```

## Install Local Python Dependencies

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Product Data

The existing ETL/vector pipeline is still supported.

1. Crawl product metadata and upload product images to R2:

```powershell
python etl/test.py
```

This writes product metadata to:

```text
data/fashion_metadata.json
```

2. Generate FashionCLIP vectors:

```powershell
python AI/vectorizer.py
```

This writes:

```text
data/fashion_with_vectors.json
```

3. Initialize PostgreSQL schema and load vectors:

If running scripts from the host against Docker Postgres, set:

```env
DB_HOST=localhost
DB_PORT=5433
DB_USER=pbtien
DB_PASSWORD=change_me
DB_NAME=fashion_db
```

Then run:

```powershell
python db/database.py
```

The DB schema stores embeddings as:

```sql
embedding VECTOR(512)
```

### Embedding Model

The project default embedding model is now FashionCLIP:

```env
EMBEDDING_MODEL=patrickjohncyh/fashion-clip
EMBEDDING_VERSION=2
```

To re-vectorize the catalog with the default FashionCLIP model:

```powershell
$env:EMBEDDING_MODEL="patrickjohncyh/fashion-clip"
$env:EMBEDDING_VERSION="2"
python AI\vectorizer.py
python db\database.py
```

Then run the API with the same env:

```powershell
$env:EMBEDDING_MODEL="patrickjohncyh/fashion-clip"
$env:EMBEDDING_VERSION="2"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For Docker Compose, these are already the defaults unless `.env` overrides them. The API filters search candidates by `embedding_model` and `embedding_version`, so query embeddings and database embeddings must match.

The database still uses:

```sql
embedding VECTOR(512)
```

If a model returns a dimension other than 512, vectorization/search fails with a clear error. Do not change the pgvector dimension without a separate migration/table plan.

To compare against the old CLIP baseline, temporarily set CLIP version 1, re-vectorize/reload, and run eval:

```powershell
$env:EMBEDDING_MODEL="openai/clip-vit-base-patch32"
$env:EMBEDDING_VERSION="1"
python AI\vectorizer.py
python db\database.py
python evaluation\run_eval.py --api-url http://localhost:8000 --labels evaluation\labels.json --run-name clip_v1

$env:EMBEDDING_MODEL="patrickjohncyh/fashion-clip"
$env:EMBEDDING_VERSION="2"
python AI\vectorizer.py
python db\database.py
python evaluation\run_eval.py --api-url http://localhost:8000 --labels evaluation\labels.json --run-name fashionclip_v2

python evaluation\compare_runs.py evaluation\results\clip_v1.json evaluation\results\fashionclip_v2.json
```

## Run Full App

Start Docker Desktop, then run:

```powershell
docker compose up --build
```

URLs:

- Frontend: [http://localhost:5173](http://localhost:5173)
- API health: [http://localhost:8000/health](http://localhost:8000/health)
- Object detection health: [http://localhost:8001/health](http://localhost:8001/health)
- Postgres host port: `localhost:5433`

## Run Pieces Locally

API:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Object detection:

```powershell
uvicorn object_detection.api:app --host 0.0.0.0 --port 8001
```

Frontend:

```powershell
cd web\fe
npm install
npm run dev
```

## Troubleshooting

### Missing YOLO model

Put the real model file at:

```text
models/object-detection/best.pt
```

The repository does not commit `.pt` files.

### DB has no vector data

If upload works but products are empty, load vector data:

```powershell
python AI/vectorizer.py
python db/database.py
```

Confirm Docker Postgres is reachable on `localhost:5433` when running host scripts.

### Embedding model download or cache issues

The first FashionCLIP vectorization loads `patrickjohncyh/fashion-clip` through Hugging Face. It may need network access and write access to the Hugging Face cache. If cache permissions fail, clear or fix permissions for:

```text
%USERPROFILE%\.cache\huggingface
```

### Frontend does not show `s3://` image URLs

The API returns:

- `image_url`: raw stored URL, including `s3://...`
- `usable_image_url`: a public `http(s)` URL when `R2_PUBLIC_BASE_URL` is configured, or an API image proxy path such as `/api/products/{id}/image`

For the simplest setup, make the R2 bucket public through a custom/public domain and set:

```env
R2_PUBLIC_BASE_URL=https://your-public-r2-domain
```

If the bucket is private, keep `R2_PUBLIC_BASE_URL` empty and configure the backend R2 credentials in `.env`; the API will proxy product images without exposing secrets to the frontend.

## Verification Checklist

```powershell
python -m compileall AI db etl api
docker compose config
cd web\fe
npm run build
```

Runtime checks:

- `GET http://localhost:8000/health`
- `GET http://localhost:8001/health`
- upload a real outfit image at `http://localhost:5173`

## Notes

- `/api/search/url` is intentionally not implemented in the MVP to avoid URL-fetch/SSRF risk.
- Product search uses FashionCLIP vectors and PostgreSQL pgvector cosine distance.
- Object detection returns raw JPEG base64 crops without a `data:image` prefix; the frontend prepends that for previews.
