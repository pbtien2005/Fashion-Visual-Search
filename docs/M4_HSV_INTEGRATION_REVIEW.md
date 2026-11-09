# M4 review: HSV147 color integration only

This is a concrete review boundary for the M4 owner. No external message was sent and no M4 approval is implied. Changes are in the working tree for review; deployment still requires the complete sidecar and live services.

Scoped patch for review: [M4_HSV_INTEGRATION.patch](M4_HSV_INTEGRATION.patch). It contains only `api/search_service.py` and its existing M4 unit tests, relative to base HEAD. API source-image forwarding and M3 cache/extraction modules are documented separately.

## Changed M4 functions

| Function in `api/search_service.py` | Integration change |
|---|---|
| `search_match_for_crop` | Accept original RGB from API; extract one raw-ROI HSV descriptor, shared across encoder variants |
| `_search_single_crop_variant` | Call `AI.features.extract_runtime_features`; preserve encoder crop input and candidate query parameters |
| `rerank_products` | Accept `query_color`; lookup catalog HSV by product ID/model/version; use Hellinger similarity as color_score; no old RGB name penalty |
| `serialize_product` | Preserve API keys; display RGB/name come from the HSV sidecar, including catalog listing; null when unavailable |
| `compute_color_penalty` | Compatibility-only symbol returning 0; not called in ranking; old RGB threshold getters removed |

## Preserved logic

- Model ID/version, encoder/processor and normalized 512-vector contract.
- pgvector cosine retrieval, model/version filtering, category candidate policy and limits.
- Image/category/text-style formulas, subtype/style inference and text-style-absence weight normalization.
- Default/configurable weights, category penalty, candidate variant merge, top-k and confidence gates.
- API response shape and frontend.

The product color label added to text prompts now comes from HSV display mapping; no legacy DB color label is used as fallback. Names are approximate because H-S discards chromatic V. This is the only color-related input change to existing text prompt construction.

## Review decisions made explicit

1. Hellinger similarity already has [0,1] range; no unverified rescaling.
2. RGB mismatch threshold/caps/penalty are not valid histogram calibration; `color_penalty=0` until M4 evaluates any HSV-specific penalty. Weights have not changed.
3. Missing/invalid cache/query color: `HSV_MISSING_POLICY=zero` contributes 0 without redistributing color weight, or `error` rejects the request. Warnings record IDs/reasons. No RGB fallback.
4. Cache config/version/model/ROI mismatch is invalid, never mixed silently.

## M4 owner acceptance checklist

- Review the scoped diff of `api/search_service.py` and `tests/test_search_service.py`; supplementary production routing tests are in `tests/test_production_hsv.py`.
- Confirm the chosen missing policy and the intentional removal of RGB-specific color penalty.
- Provide complete catalog sidecar matching existing DB IDs/model/version/ROI policy.
- Run tests and staging HTTP checks using live YOLO/DB; check metadata/coverage and existing confidence gates.
- Calibrate future HSV score policies only with independent validation data; no ranking weight tuning is bundled here.

M3 extraction/cache/deployment instructions: [M3_HSV_PRODUCTION.md](M3_HSV_PRODUCTION.md).
