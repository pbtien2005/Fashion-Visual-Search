# M3 — Feature Extraction: bàn giao độc lập

> Cập nhật sau quyết định chuyển production sang HSV147: xem [M3_HSV_PRODUCTION.md](M3_HSV_PRODUCTION.md) và [M4_HSV_INTEGRATION_REVIEW.md](M4_HSV_INTEGRATION_REVIEW.md). Các nhận xét “API chưa nối HSV” bên dưới là snapshot của giai đoạn handoff trước, không mô tả code production sau cập nhật.

Ngày kiểm chứng: 2026-11-08. Base branch `vinh`, commit `5a433e9e55201c89f5b3fa25e2dc6339d6ba66b2`; thay đổi handoff nằm trên working tree sau commit này. Báo cáo inference ghi source hashes để xác định đúng code đã chạy.

## 1. Trạng thái và ranh giới bàn giao

M3 cung cấp extractor HSV độc lập, FeatureBundle có validation, adapter input M2, cache màu catalog, lookup có kiểm tra tương thích và smoke check pretrained FashionCLIP. Không cần website, YOLO hay database để dùng các chức năng này.

Đã chạy **FashionCLIP với weights thật**, không mock encoder: model `patrickjohncyh/fashion-clip`, embedding version `2`, Hugging Face revision `7e3ba62ce16b379a1ab479346b66f192e76f51b7`. Embedding 512 chiều, L2 norm khoảng `0.99999997145`; HSV147 L1=1; bundle JSON round-trip và query/catalog cùng ảnh/config cho descriptor giống nhau. Ảnh thử là ảnh starter catalog, chưa phải source ROI quần áo được M2 xác nhận. Mask trong lần inference này thiếu và được ghi `mask_missing`; mask đúng/sai được kiểm tra riêng bằng fixture.

**Không sửa code M2/M4, thuật toán HSV, model/processor FashionCLIP hoặc ranking weights.** Không chạy benchmark chất lượng retrieval mới. API production hiện chưa gọi interface HSV; việc nối HTTP/reranking thuộc M2/M4. Phạm vi hoàn thành ở đây là thư viện M3 có contract và kiểm chứng để bàn giao độc lập.

## 2. File bàn giao

| File | Vai trò |
|---|---|
| `AI/features.py` | `FeatureBundle` (alias tương thích của `M3Features`), validation/JSON round-trip, adapter detection và extract kết hợp |
| `AI/hsv_descriptor.py` | HSV144/147 và histogram similarity hiện có, giữ nguyên |
| `AI/embedding_model.py` | FashionCLIP encoder/processor hiện có, giữ nguyên |
| `AI/color_cache.py` | Batch cache màu riêng; reader và `catalog_descriptor_for_query()` kiểm tra query/catalog |
| `AI/verify_m3.py` | Smoke check weights thật, bundle/cache round-trip; mặc định offline |
| `docs/m3_contract/hsv147.json` | Profile dùng chung cho query/catalog; không tự suy ra cấu hình từ tên file |
| `docs/m3_contract/m2_input.schema.json` | JSON Schema payload mà adapter M3 nhận; không phải sửa schema HTTP của M2 |
| `docs/m3_contract/feature_bundle.schema.json` | JSON Schema output cho profile 144/147 chiều |
| `tests/test_m3_handoff.py` | Test contract, config/policy/version mismatch, mask coordinates, batch CLI và bảo vệ file đã tồn tại |
| `docs/m3_handoff_evidence/` | `report.json`, `bundle.json`, `color_cache.json` từ weights thật; bundle chứa đủ 512/147 số, không dùng vector giả |

Schema mô tả cấu trúc; các điều kiện finite, L2/L1, version/config và pixel count phải kiểm tra thêm bằng runtime validator. Không dùng riêng JSON Schema để kết luận vector hợp lệ.

## 3. Version và cấu hình dùng chung

Ba version độc lập:

- Bundle schema: **`m3-features-v1`**.
- Embedding: model **`patrickjohncyh/fashion-clip`**, tracking version **`2`**, dimension **512**. Tracking version 2 không thay thế revision weights; revision thực tế được lưu trong smoke report.
- Color profile đi kèm: **`hsv-hs-v1-147-b0681e3cbf842188`**. Giá trị lấy từ `HSVConfig.version`, không hardcode hash khi tạo descriptor.

Profile `hsv147.json`: H bins=18, S bins=8, achromatic=true, S threshold=32, black V max=40, white V min=220; `mask_policy=roi`, min fraction=.001, max fraction=1, min pixels=1. H range [0,180), S range [0,256), OpenCV RGB→HSV, L1 normalization. Các ngưỡng là giá trị khởi đầu chưa tối ưu bằng validation.

HSV144 dùng `dataclasses.replace(config, achromatic=False)` và phải tạo cache riêng. Bins/ngưỡng/mask policy khác nhau cũng sinh color version khác nhau. Query và catalog phải load **cùng profile**, cùng model/version và cùng ROI policy. Cùng config không chứng minh ROI có cùng chất lượng: M4 vẫn phải kiểm tra provenance/fallback.

## 4. Schema M2 → M3

### Input Python trực tiếp

```python
extract_features(
    garment_id,        # ID không rỗng; dùng cùng product/garment identity khi liên kết metadata
    embedding_image,   # PIL RGB hoặc ảnh RGB mà encoder hiện tại đã nhận
    roi_rgb,           # numpy.ndarray uint8 H×W×3, RGB, không rỗng
    mask=None,         # ndarray bool/uint8 H×W; uint8 chỉ có 0/1/255
    config=None,       # HSVConfig; handoff yêu cầu truyền config đã load từ profile chung
    metadata=None,     # dict; nên có roi_source và roi_policy
)
```

`roi_rgb` dành cho màu là ROI nguồn chưa gamma, chưa padding trắng, chưa resize cho encoder. `embedding_image` giữ đúng input policy encoder hiện tại; M3 không sửa preprocessing/processor. Nếu nguồn từ OpenCV BGR, caller phải đổi `cv2.COLOR_BGR2RGB` đúng một lần. Không thể suy ra RGB/BGR chỉ từ shape/dtype.

Mask phải khớp **cả kích thước lẫn hệ tọa độ** ROI. M3 kiểm tra shape/dtype/giá trị; không thể phát hiện mask bị dịch vị trí nhưng vẫn cùng H×W. Full-image mask cần được M2 cắt bằng cùng bbox trước khi chuyển sang M3. M3 không tự resize/warp mask hoặc suy đoán foreground.

### Adapter payload từ M2

```python
extract_detection_features(garment_id, detection, original_rgb=None, config=None)
```

| Trường | Yêu cầu / ý nghĩa |
|---|---|
| `image_base64` | Bắt buộc khi tạo bundle: crop JPEG/PNG RGB dành cho encoder hiện tại; plain base64 hoặc image data URL |
| `source_roi_base64` | Tùy chọn, ưu tiên cao nhất cho màu; đề nghị PNG lossless ROI nguồn; plain base64 hoặc image data URL |
| `mask_base64` | Tùy chọn, **plain base64 PNG** single-channel binary, cùng tọa độ ROI được chọn; không dùng data URL cho mask |
| `original_bbox` / `bbox` | `[x1,y1,x2,y2]`, integer, trên ảnh gốc, x2/y2 exclusive; cần nếu dùng `original_rgb` và không có source ROI payload |
| `class_name` | Tùy chọn; được chuyển vào metadata |
| `original_rgb` | Đối số Python riêng, RGB uint8 ảnh nguồn; không phải trường JSON/base64 tự động của adapter |

Thứ tự chọn nguồn màu:

1. `source_roi_base64` hợp lệ → `roi_source=source_roi_base64`.
2. Nếu không có trường trên và có `original_rgb`, cắt bằng `original_bbox` hoặc `bbox` → `original_image_bbox`.
3. Nếu không có cả hai, decode crop cũ → `legacy_processed_crop`, metadata `roi_fallback_reason=source_roi_unavailable`.

Nguồn được cung cấp nhưng ảnh/bbox sai sẽ **báo lỗi**, không âm thầm chuyển sang nguồn khác. ROI policy mặc định của adapter là `source_roi_v1`; đây là quy tắc ưu tiên nguồn/fallback nêu trên. Khi cần policy khác, dùng API Python trực tiếp và đặt cùng policy trong manifest catalog.

### Ví dụ input và gọi adapter

```python
import base64
import json
from pathlib import Path
from AI.features import extract_detection_features
from AI.hsv_descriptor import HSVConfig

config = HSVConfig(**json.loads(Path("docs/m3_contract/hsv147.json").read_text()))
detection = {
    "class_name": "short_sleeve_top",
    "bbox": [40, 30, 240, 330],
    "image_base64": base64.b64encode(Path("encoder_crop.jpg").read_bytes()).decode(),
    "source_roi_base64": base64.b64encode(Path("source_roi.png").read_bytes()).decode(),
    "mask_base64": base64.b64encode(Path("roi_mask.png").read_bytes()).decode(),
}
bundle = extract_detection_features("garment-001", detection, config=config)
payload = bundle.to_dict()
```

Ví dụ này dùng file do caller/M2 cung cấp, chưa có các file production đó trong checkout. Schema M2 hiện tại chưa phát các trường ROI/mask trên; M2 có thể tích hợp sau mà không cần thay thuật toán M3.

## 5. Schema M3 → M4

`FeatureBundle` và `M3Features` là cùng một dataclass. Output `.to_dict()`:

| Trường | Nội dung |
|---|---|
| `schema_version` | `m3-features-v1` |
| `garment_id` | ID string không rỗng |
| `embedding` | 512 float hữu hạn, norm L2≈1; không có fake vector khi encoder lỗi |
| `embedding_model`, `embedding_version` | Model identity và tracking version |
| `color.values` | 144 hoặc 147 float không âm; L1=1 khi có pixel, zero nếu policy empty không có pixel |
| `color.version`, `color.config` | Full descriptor identity/spec; config gồm bins/ranges/normalization/thresholds/policy |
| `color.roi_source`, `color.roi_policy` | Nguồn thực tế và policy xử lý |
| `color.mask_status`, `color.fallback_reason` | Trạng thái và lý do fallback |
| `color.valid_pixels`, `total_pixels`, `mask_fraction` | Pixel coverage; mask_fraction null khi không đo được mask |
| `color.metadata`, `metadata` | Provenance bổ sung; không đặt ndarray/PIL vào metadata nếu cần JSON |

Ví dụ output thật, đầy đủ các số: [bundle.json](m3_handoff_evidence/bundle.json). File này là kết quả smoke trên ảnh starter, không phải sản phẩm đã được M2 phát hiện. Các số kiểm chứng tóm tắt nằm trong [report.json](m3_handoff_evidence/report.json).

M4 nhận bundle JSON:

```python
import json
from pathlib import Path
from AI.features import FeatureBundle
from AI.color_cache import catalog_descriptor_for_query
from AI.hsv_descriptor import histogram_similarity

query = FeatureBundle.from_dict(json.loads(Path("query_bundle.json").read_text()))
cache = json.loads(Path("catalog_hsv.json").read_text())
product_color = catalog_descriptor_for_query(
    cache, "product-001", query.color, query.embedding_model, query.embedding_version
)
color_score = histogram_similarity(query.color, product_color)
# Candidate retrieval, ranking weights, score calibration và policy bỏ qua/phạt thiếu màu thuộc M4.
```

Lookup kiểm tra cache schema, config, model/version, ROI policy, color version, key integrity và descriptor validity. Thiếu product ID → `KeyError`; mismatch hoặc nhiều entry cùng product → `ValueError`. M3 không tự chuyển sang descriptor khác version, không giả lập score khi thiếu cache. M4 quyết định xử lý các lỗi này trong ranking; không chuyển trực tiếp ngưỡng/penalty hiện có sang histogram.

`histogram_similarity()` là Hellinger similarity `1-sqrt(max(0,1-sum(sqrt(p*q))))`, hoặc Intersection khi truyền `metric="intersection"`. Empty ở một phía trả 0. Similarity là score, không phải xác suất đúng sản phẩm. Histogram so sánh version/config; ROI policy được kiểm tra thêm tại cache lookup.

## 6. Fallback và lỗi

| Trường hợp | Hành vi |
|---|---|
| ROI rỗng/sai dtype/shape hoặc ảnh base64 sai | `ValueError`, không có bundle giả |
| Mask thiếu | `mask_status=missing`, `fallback_reason=mask_missing` |
| Mask decode lỗi, không phải PNG, shape/dtype/value sai | `mask_status=invalid`, `mask_invalid` |
| Mask không có foreground | `empty`, `mask_empty` |
| Mask dưới min pixels/fraction hoặc trên max fraction | `insufficient`, `mask_insufficient` |
| `mask_policy=roi` | Dùng toàn ROI khi mask không hợp lệ; giữ các pixel trắng |
| `mask_policy=empty` | Trả histogram zero, valid_pixels=0; cache builder ghi failure, không cache descriptor giả |
| `mask_policy=error` | Mask không hợp lệ làm extraction lỗi; cache builder ghi failure |
| Chỉ có crop encoder cũ | Ghi legacy source và ROI fallback; mask thiếu vẫn được ghi riêng |
| Encoder/load weights lỗi hoặc vector invalid | Exception truyền lên caller; không xuất bundle giả |
| Catalog thiếu ảnh | Failure có product ID; không thay bằng descriptor giả |

Bundle không rỗng về ảnh không có nghĩa descriptor màu có pixel: M4 phải đọc valid_pixels. Fallback có thể đưa background vào histogram và không được coi như segmentation thật.

## 7. Tạo cache toàn bộ catalog

Cache màu không cần load FashionCLIP, DB hoặc YOLO; model/version trong cache là identity để liên kết với embedding đã tồn tại, không chứng minh encoder đã chạy trên mỗi sản phẩm. Không tạo lại embeddings hoặc ghi catalog gốc.

### Chuẩn bị manifest

Stage toàn bộ ảnh ROI nguồn và mask (nếu có) thành local files. CLI không tự tải HTTP/R2 hoặc detect quần áo. Với ảnh chỉ có full-product photo, ghi `roi_source` trung thực và dùng policy phù hợp cả query/catalog.

Ví dụ `data/m3_manifest.json`:

```json
{
  "roi_policy": "source_roi_v1",
  "catalog": [
    {"id": "product-001", "image": "rois/product-001.png", "mask": "masks/product-001.png", "roi_source": "m2_source_roi"},
    {"id": "product-002", "image": "rois/product-002.png", "roi_source": "m2_source_roi"}
  ]
}
```

Paths tương đối với thư mục manifest, không phải cwd. ID bắt buộc, không trùng; mỗi product một entry trong mỗi cache. Mask phải theo đúng ROI tương ứng. Thiếu ảnh vẫn để entry trong manifest để failure/coverage phản ánh tổng catalog. Không bỏ âm thầm sản phẩm lỗi.

### Lệnh batch

Từ repository root, thư mục output cha đã tồn tại, tên file output mới:

```powershell
.venv\Scripts\python -B -m AI.color_cache --manifest data/m3_manifest.json --config docs/m3_contract/hsv147.json --model-name patrickjohncyh/fashion-clip --model-version 2 --output output/catalog_hsv147_run001.json
```

Test CLI trên ảnh mẫu có sẵn, không phải toàn bộ catalog (manifest mẫu dùng `whole_catalog_photo_v1`, vì là ảnh sản phẩm nguyên gốc; không ghép cache này với query policy `source_roi_v1`):

```powershell
.venv\Scripts\python -B -m AI.color_cache --manifest docs/m3_results/catalog_manifest.json --config docs/m3_contract/hsv147.json --output output/catalog_sample_handoff_run001.json
```

CLI từ chối ghi đè file đã tồn tại. Sau chạy, đọc `coverage.total/extracted/fraction`, `mask_status_counts`, `failures`. CLI có thể exit 0 khi có failures từng sản phẩm; **người vận hành phải kiểm tra failures/coverage**, không dùng riêng exit code làm bằng chứng đủ catalog. Đây là JSON cache in-memory, chưa có streaming/resume: với catalog lớn, chia manifest thành các shard ID không chồng nhau, giữ cùng config/model/policy; lưu từng cache riêng và để consumer lập index theo shard. Helper lookup mẫu quét entries, nên M4 nên lập product-ID→cache-key index một lần khi load catalog thay vì quét lại cho mỗi candidate.

Query dùng cùng profile:

```python
config = HSVConfig(**json.loads(Path("docs/m3_contract/hsv147.json").read_text()))
bundle = extract_features(product_id, encoder_image, roi_rgb, mask, config,
    {"roi_source": "m2_source_roi", "roi_policy": "source_roi_v1"})
```

## 8. Dependencies, weights và tái lập

Môi trường kiểm chứng thật: Python 3.12.10, Torch 2.7.1+cpu, Transformers 4.57.1, NumPy 2.1.3, OpenCV 5.0.0, Pillow 11.3.0. Môi trường có sẵn dùng system-site-packages; **chưa xác minh trên đúng toàn bộ pins trong requirements.txt**, đặc biệt Torch 2.11.0/Transformers 4.57.0. Không thay pins, processor hay chọn `use_fast` khác trong lần bàn giao này.

Máy mới cần cài dependencies dự án hoặc chuẩn bị môi trường tương thích. HSV/cache độc lập chỉ cần NumPy, OpenCV, Pillow. Bundle đầy đủ cần Torch, Transformers và python-dotenv; tải weights cần huggingface_hub (dependency của Transformers), mạng hoặc cache được copy từ máy khác. Weights không commit, nằm dưới `.hf_cache/hub` được Git ignore.

Để tải đúng snapshot đã kiểm chứng, có thể chạy Python sau (cần mạng và đủ dung lượng cho weights):

```python
from huggingface_hub import snapshot_download
snapshot_download("patrickjohncyh/fashion-clip", revision="7e3ba62ce16b379a1ab479346b66f192e76f51b7",
    cache_dir=".hf_cache/hub", allow_patterns=["config.json", "preprocessor_config.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "model.safetensors"])
```

Loader hiện tại dùng model ID với ref `main`, không có tham số revision. Để tái lập offline đúng snapshot trên máy mới, sau tải pinned snapshot, tạo `.hf_cache/hub/models--patrickjohncyh--fashion-clip/refs/main` với nội dung revision trên **chỉ trong cache mới dành riêng lần chạy**; hoặc copy toàn bộ cache đã kiểm chứng. Không sửa ref của cache dùng chung đang phục vụ công việc khác. `--allow-download` có thể lấy main mới nên phải kiểm tra revision trong report trước khi so kết quả.

Smoke check offline (tên output directory phải mới):

```powershell
.venv\Scripts\python -B -m AI.verify_m3 --image evaluation/queries/starter_catalog/B0GGYQ4J5Z.jpg --config docs/m3_contract/hsv147.json --cache-dir .hf_cache/hub --output output/m3_real_weights_run002
```

Thêm `--mask path/to/roi_mask.png` khi có mask đúng; thêm `--allow-download` để cho loader tải khi cache chưa có. Script không thay model, chỉ chạy encoder hiện có, kiểm tra bundle round-trip, tạo cache **một ảnh** trong output mới và kiểm tra query/catalog identical input. Report ghi revision, weights SHA256, environment, source/input hashes; failures được ghi `status=failed`, process exit 1. Không gọi đây là benchmark retrieval hoặc cache toàn catalog.

## 9. Lệnh test và kết quả

```powershell
# Tests handoff độc lập, không tải weights
.venv\Scripts\python -B -m unittest discover -s tests -p test_m3_handoff.py -v
# Toàn bộ suite hồi quy hiện có
.venv\Scripts\python -B -m unittest discover -s tests -v
git diff --check
```

Bộ test hiện có gồm test bundle với mock encoder và test tensor normalization với model giả; các test đó không thay thế smoke weights thật bên trên. Test suite hiện có chứa benchmark-helper unit tests; không chạy script benchmark chất lượng mới và không lập báo cáo so sánh RGB trong lần handoff này.

Kết quả cuối: **54 pass, 0 fail, 0 error**, gồm 47 test có sẵn và 7 test handoff bổ sung. Log tại [tests.txt](m3_handoff_evidence/tests.txt); unittest báo 2.764 giây thực thi, không tính toàn bộ thời gian import dependencies. `git diff --check` pass. Hai JSON Schema đã được `Draft202012Validator.check_schema()` kiểm tra; output bundle thật và payload ảnh PNG fixture đều validate thành công. CLI integration test tự tạo/xóa ảnh fixture trong `output/`, xác minh missing-image failure và từ chối overwrite. Smoke weights thật đã chạy lại sau thay đổi source cuối, source hashes trong report khớp working tree.

## 10. Phần bàn giao cho M2/M4 và phần chưa kiểm chứng

| Hạng mục | Trạng thái / bên nhận |
|---|---|
| Weights thật, vector512/L2, bundle147/JSON, query/catalog cùng config | Đã chạy độc lập trên ảnh starter; xem evidence |
| Mask PNG, shape/value, giữ foreground trắng, fallback | Đã kiểm tra fixture; chưa có mask quần áo thật từ M2 |
| Source ROI contract | Đã có adapter/schema; M2 cần cung cấp ROI chưa gamma/padding/resize và mask đúng tọa độ |
| Full catalog | Chưa có `data/`/catalog production; cung cấp manifest và ảnh để vận hành batch theo mục 7 |
| M4 consumer | Nhận `FeatureBundle`, cache reader và policy missing/mismatch; M4 tự nối candidate/reranking, không cần đổi M3 |
| HTTP M2→M3 và endpoint tìm kiếm HSV | Chưa kiểm chứng/chưa nối production; không sửa M2/M4 trong lần này |
| Chất lượng retrieval, calibration, latency toàn pipeline | Chưa đánh giá; không có benchmark mới, không khẳng định cải thiện |

M2 và M4 có thể tích hợp độc lập theo contract này; các dependencies production còn thiếu không ngăn bàn giao thư viện M3. Cần giữ các hạn chế trên khi báo cáo tiến độ, không gọi smoke một ảnh là end-to-end retrieval.
