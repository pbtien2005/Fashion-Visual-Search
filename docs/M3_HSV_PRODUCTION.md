# Production migration: FashionCLIP + HSV147

Base HEAD: `5a433e9e55201c89f5b3fa25e2dc6339d6ba66b2`, branch `vinh`. Existing handoff changes in the working tree were preserved. This change activates HSV in runtime code; the earlier independent-module reports remain historical snapshots.

## Runtime flow

1. `/api/search/upload` retains the uploaded RGB image after the unchanged detection call. It uses the same OpenCV decode/orientation path as M2, followed by explicit BGR→RGB, so JPEG EXIF rotation cannot silently move the bbox coordinate frame.
2. `extract_source_color()` selects source ROI payload, otherwise original-image bbox, otherwise explicitly recorded legacy crop. It uses the existing HSV147 extractor unchanged. A failed supplied ROI is unavailable, never RGB fallback.
3. One query color descriptor is reused across tight/padded/wide encoder variants. `AI.features.extract_runtime_features()` runs the unchanged FashionCLIP encoder on each existing encoder crop. A valid descriptor yields the existing `M3Features` bundle; unavailable color yields an explicit `EmbeddingOnlyFeatures` result, not a fabricated histogram.
4. Candidate retrieval remains model/version-filtered FashionCLIP/pgvector cosine search, same limits and category policy.
5. M4 looks up each candidate in the versioned HSV sidecar, validates model/config/color version/ROI policy, and uses existing Hellinger similarity directly in [0,1].
6. Color display fields keep the same API names and types. Their values now come from HSV; there are no RGB extraction or RGB distance calls in the runtime path.

## Ownership

- M3: `AI/production_color.py`, `features.py`, `vectorizer.py`, `color_cache.py`, `rebuild_color_cache.py`, shared profile and extraction/cache tests.
- M4 review boundary: `api/search_service.py` replaces only color preparation/lookup/similarity and derives color labels before existing text/style scoring. See [M4_HSV_INTEGRATION_REVIEW.md](M4_HSV_INTEGRATION_REVIEW.md). Category/image/text-style formulas, candidate limits, merging, match gates and weight configuration remain unchanged.
- API glue: `api/main.py` passes the original upload RGB into search; schemas and frontend files are unchanged.
- DB compatibility: `db/database.py` retains the same columns and SQL structure. New display RGB/name metadata is derived from the HSV payload on import. Runtime search/list serialization overlays cached HSV display metadata, so old DB color values cannot re-enter scoring or display as a fallback.
- Maintenance compatibility: `etl/backfill_product_colors.py` forwards to the color-only rebuild CLI (explicit catalog/output arguments); `db/backfill_color_names.py` refreshes display fields from the HSV sidecar instead of deriving names from legacy DB RGB.
- YOLO and `AI/hsv_descriptor.py`/`AI/embedding_model.py` are unchanged in this migration.

## Official config and cache identity

Default `HSVConfig()` is the official 18×8 + Black/Gray/White profile (147 dimensions). `HSV_CONFIG_PATH` optionally points to the same JSON profile for both query and catalog. Runtime rejects a non-147 profile. Shared file: `docs/m3_contract/hsv147.json`.

- Color version of supplied profile: `hsv-hs-v1-147-b0681e3cbf842188`.
- ROI policy: `source_roi_v1`; same priority source ROI → original bbox → explicit fallback.
- `HSV_COLOR_CACHE_PATH`: default `data/color_cache_hsv147.json`.
- Cache keys retain product ID, embedding model/version, ROI policy, color version and source hashes.
- Model identity/version still `patrickjohncyh/fashion-clip`, 2; embedding 512 and L2 unchanged.
- Store builds a product index once, reloads when path/mtime/size/config changes, and validates candidate descriptors on access. Publish caches atomically as new files; restart Docker after switching a bind-mounted file.

Catalog raw photographs use the same source-selection helper. `catalog_color_inputs()` calls the existing detector to obtain a bbox; no new detector model/algorithm is introduced. The crop URL resolver now respects configured `OBJECT_DETECTION_URL`, including Docker service names; `PRODUCT_CROP_DETECTION_URL` still overrides it. If detection is unavailable, original whole image is the recorded fallback. `USE_PRODUCT_CROP_EMBEDDING` still controls only encoder cropping and keeps its old default false. Color ROI selection is independent of this flag.

Same policy/config does not guarantee equally good ROI: whole-image fallback may contain background. Inspect descriptor source/mask metadata and cache coverage before deployment. Masks follow existing configured ROI/empty/error behavior; no RGB fallback exists for any failure.

## Missing/invalid policy and score scale

`HSV_MISSING_POLICY=zero` (default): log a warning with item ID/reason; unavailable/invalid/empty color contributes exactly 0. Color weight is not redistributed; existing text/style-absence normalization remains unchanged. Encoder failures continue to fail extraction; they are not hidden as color failures.

`HSV_MISSING_POLICY=error`: reject the request/extraction on missing/invalid query or catalog descriptor. Use this for a fully populated deployment when silent partial color coverage is unacceptable. Configuration errors fail before extraction. Failures never invoke Dominant RGB.

Hellinger similarity `1-sqrt(1-BC)` already lies in [0,1], so no unmeasured scale transform has been added. Weights stay `.60/.15/.15/.10` (image/category/color/text-style), or the existing normalized environment values. Old RGB name mismatch caps/thresholds/penalties have no calibration for histogram scores and are not reused; API `color_penalty` remains present with value 0. `compute_color_penalty()` remains only as a compatibility symbol returning 0. Calibration of any future HSV penalty requires M4 validation and review.

## Frontend compatibility

No changes to response schemas or frontend. Existing fields `query_color_rgb/name`, `dominant_color_rgb/name`, `color_score`, `color_penalty` remain.

RGB fields are **display swatches**, not RGB feature extraction. `descriptor_display()` selects the largest histogram bin/achromatic group. Black/Gray/White have fixed swatches; chromatic H/S bin centers are rendered at V=192 and labeled by hue sector. HSV H-S does not retain chromatic V, so navy/brown/tan cannot be faithfully reconstructed. Labels are approximate. Missing descriptors yield null swatch/name, never stale DB RGB/name. Name-based caps do not affect histogram distance. Existing product text/style construction can now include the HSV-derived display label; its formula and weights are unchanged.

## Rebuild catalog colors without re-embedding

Original catalog and old caches stay untouched. Prepare a JSON list containing the same product IDs as PostgreSQL. Local images can use `image_path` or `image`, relative to catalog JSON; remote inputs use existing `r2_image_url`/`thumbnailImage`/`image_url`. An existing M2 payload may be staged as `detection` on a record. This rebuild does not call the encoder or write embeddings.

```powershell
# Local full catalog, existing M2 detection service for source bboxes
.venv\Scripts\python -B -m AI.rebuild_color_cache --catalog data/fashion_with_vectors.json --config docs/m3_contract/hsv147.json --output data/color_cache_hsv147_run001.json

# Explicitly authorize HTTP/R2 fetching when catalog rows have only remote URLs
.venv\Scripts\python -B -m AI.rebuild_color_cache --catalog data/fashion_with_vectors.json --config docs/m3_contract/hsv147.json --allow-remote --output data/color_cache_hsv147_run002.json
```

`--skip-detection` deliberately uses whole-original-image fallback (or a staged detection if present); it is recorded in ROI metadata. It is useful for already-cropped images or service-independent preparation, but is not evidence of real garment segmentation.

For a prepared local ROI/mask manifest, existing command remains available:

```powershell
.venv\Scripts\python -B -m AI.color_cache --manifest data/m3_roi_manifest.json --config docs/m3_contract/hsv147.json --output data/color_cache_hsv147_run003.json
```

Its manifest must specify `roi_policy: source_roi_v1`; images must be source ROIs. Old sample caches using `whole_catalog_photo_v1` are intentionally rejected by production. Rebuild them under the actual policy instead of relabeling old cache identity.

Both commands require a new output file and existing parent directory. Check `coverage`, `failures`, `mask_status_counts` and ROI fallback metadata. The production rebuild exits 2 if any product failed, while preserving a diagnostic partial cache. Duplicate/missing IDs fail the job. The generic prepared-ROI CLI keeps its existing exit behavior; inspect coverage even when exit code is 0. For large catalogs, split into disjoint manifests and provision a validated complete sidecar; JSON building is in memory.

The normal vectorization pipeline now also uses HSV and writes new artifacts by default: `data/fashion_with_hsv_vectors.json` and `data/color_cache_hsv147.json`. It refuses to overwrite either. The default DB loader now reads the new vector artifact; explicit input paths remain supported. Use the color-only rebuild for existing embeddings; do not invoke full vectorization just to refresh color.

## Configure and deploy

```powershell
$env:HSV_CONFIG_PATH = 'docs/m3_contract/hsv147.json'
$env:HSV_COLOR_CACHE_PATH = 'data/color_cache_hsv147_run001.json'
$env:HSV_MISSING_POLICY = 'error'
# Start the normal API after weights, DB, YOLO service and the complete cache are available.
```

Docker copies the same profile into `/app/config/hsv147.json`, binds the host cache read-only at `/app/color-cache/catalog.json`, and does not auto-create a missing cache path. Set `HSV_COLOR_CACHE_FILE` to the host artifact before `docker compose up --build`; missing file prevents accidental startup with a fabricated directory. Schema/pgvector migration is not required. Existing DB product IDs and cache IDs must match.

## Tests and actual limits

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v

$env:HF_HUB_CACHE = (Resolve-Path '.hf_cache/hub').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:RUN_REAL_HSV_INTEGRATION = '1'
.venv\Scripts\python -B -m unittest discover -s tests -p test_production_hsv_real.py -v
```

Production tests exercise real API request/response serialization via in-process ASGI HTTP, catalog preparation and reranking, and replace legacy RGB functions with exceptions to ensure none are called. Encoder/M2/DB fixtures are used in normal unit/integration tests. The opt-in real test uses pretrained FashionCLIP for catalog and query, a real on-disk HSV sidecar and actual API handlers; only M2 transport and DB candidate transport are staged/mocked. This is not proof of live YOLO or live PostgreSQL retrieval, nor a quality benchmark.

`AI/color_utils.py` remains for historical tests/evaluation. No API/vectorizer/DB/maintenance module imports it anymore. Source was not deleted while tests and the historical color evaluator still depend on it; the evaluator was not run for a new RGB comparison. Existing maintenance entry points now route through HSV.

Live deployment remains dependent on full catalog images/cache, running PostgreSQL, YOLO weights/service and Docker engine. No production data was invented, no original cache overwritten, no RGB comparison benchmark run.

## Final verification

- **72 tests passed, 0 failed, 0 skipped**, including opt-in real pretrained FashionCLIP catalog/query HTTP integration, EXIF coordinate alignment and configured detector URL handling. Final suite execution was 31.126 seconds excluding initial imports. Full [test log](m3_production_evidence/tests.txt).
- The test environment used offline local weights; no model/processor substitutions. Only external M2/DB transport fixtures replace unavailable live services. RGB extraction/similarity/name functions were guarded with exceptions in production routing tests.
- Color-only rebuild processed **28/28 local starter images**, 0 failures; all descriptors validated through the runtime cache reader. Detection was explicitly skipped, with original whole-image fallback and missing masks recorded. This is a sample artifact, not full production coverage or real segmentation validation.
- `git diff --check`, `docker compose config --quiet` and scoped M4 patch reverse-apply check passed. HSV extractor, embedding model/processor module, YOLO, API schemas, frontend and previous cached artifacts have no diff from base HEAD.
- Windows sandbox blocked asyncio's local socketpair, so HTTP/real-weight tests ran outside the sandbox; no implementation workaround was added for that host restriction.
- Docker daemon was unavailable at check time. No live deployment, real PostgreSQL query or real YOLO inference was claimed.

Machine-readable commands, source hashes, cache coverage and limitations: [validation.json](m3_production_evidence/validation.json). M4 review: [scoped patch](M4_HSV_INTEGRATION.patch).
