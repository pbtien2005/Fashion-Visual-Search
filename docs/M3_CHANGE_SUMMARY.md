# Tổng hợp các thay đổi Module M3 – Feature Extraction

> Đây là tổng hợp tại mốc bàn giao độc lập. Phần chuyển pipeline production sang HSV147 được ghi tiếp trong [M3_HSV_PRODUCTION.md](M3_HSV_PRODUCTION.md).

Ngày tổng hợp: 08/11/2026.

Tài liệu ghi lại toàn bộ công việc M3 đã thực hiện trong đợt nâng cấp HSV Histogram, từ audit source ban đầu đến hoàn thiện thư viện và contract bàn giao cho M2/M4. Các kết quả dưới đây lấy từ code và bằng chứng chạy đã lưu; việc viết tài liệu này không chạy thêm benchmark hoặc sửa thuật toán.

## 1. Mục tiêu và phạm vi

- Bổ sung đặc trưng màu HSV Histogram, có thể chạy độc lập.
- Giữ FashionCLIP để trích xuất embedding ngữ nghĩa 512 chiều.
- Tách nguồn ảnh trích màu khỏi ảnh đã padding/resize dành cho encoder.
- Chuẩn hóa dữ liệu đầu vào, FeatureBundle đầu ra và cache màu catalog.
- Có tests, công cụ kiểm chứng và hướng dẫn bàn giao có thể tái lập.

Không viết lại YOLO, database hoặc frontend; không đổi encoder/processor FashionCLIP, không fine-tune, không tự thay ranking weights, không migration pgvector hoặc tạo lại toàn bộ embedding.

## 2. Mốc repository

| Mốc | Trạng thái |
|---|---|
| Bắt đầu đọc source | Nhánh `main`, commit `3247a5b33a016fd064ac9851410e7ec755d5e320`, working tree sạch |
| Commit chứa triển khai M3 ban đầu | `5a433e9e55201c89f5b3fa25e2dc6339d6ba66b2` – `Add M3 improvements and tests` |
| Checkout khi hoàn thiện bàn giao | Nhánh `vinh`, HEAD `5a433e9e55201c89f5b3fa25e2dc6339d6ba66b2` |
| Phần hoàn thiện contract/handoff | Nằm trong working tree sau commit trên, chưa commit tại thời điểm tổng hợp |

Không reset hoặc ghi đè thay đổi người dùng. Phân biệt rõ commit HEAD với source đang có thay đổi: report weights thật lưu source hashes của code đã chạy.

## 3. Audit source trước triển khai

Đã đọc các file AI, detection, API, DB, schemas, configuration, evaluation và tests liên quan. Ghi lại:

- Cơ chế màu trước nâng cấp là dominant RGB, chưa có HSV Histogram.
- Lọc pixel sáng có thể loại màu áo trắng và chọn vùng da còn lại.
- Query dùng crop padding/resize; catalog thường dùng ảnh nguyên gốc nên chưa thống nhất ROI policy.
- M2 chưa phát source ROI hoặc segmentation mask trong HTTP response hiện tại.
- M4 chưa có đường nhận histogram hoặc đọc cache màu.
- Các evaluator có giới hạn về nhãn độc lập, relevance sản phẩm và cách tính metric.
- Checkout không có toàn bộ catalog/vector cache, weights và mask ground truth khi bắt đầu.

Tài liệu: [M3_AUDIT.md](M3_AUDIT.md).

## 4. Triển khai HSV Histogram

File mới: [AI/hsv_descriptor.py](../AI/hsv_descriptor.py).

### 4.1. Descriptor cơ bản

- Nhận `roi_rgb`: NumPy RGB uint8, shape H×W×3, không rỗng.
- Chuyển RGB sang HSV bằng OpenCV.
- Hue 18 bins, Saturation 8 bins: **144 chiều**.
- H range `[0,180)`, S range `[0,256)`.
- Chuẩn hóa L1; tổng histogram bằng 1 khi có pixel hợp lệ.
- Lưu bins, ranges, normalization, config và descriptor version.

### 4.2. Biến thể xử lý màu vô sắc

- Thêm Black, Gray và White: **147 chiều**.
- Ngưỡng khởi đầu trên thang OpenCV uint8:
  - Vô sắc: `S <= 32`.
  - Black: vô sắc và `V <= 40`.
  - White: vô sắc và `V >= 220`.
  - Gray: vô sắc và `40 < V < 220`.
- Các pixel còn lại đi vào H-S; mỗi pixel chỉ được đếm một lần.
- Pixel tối nhưng còn bão hòa vẫn giữ nhánh có sắc.
- Không loại tất cả pixel trắng hoặc sáng khỏi ROI.

18×8 và các ngưỡng trên là cấu hình khởi đầu, chưa được chứng minh tối ưu bằng validation.

### 4.3. Mask và đầu vào lỗi

- Hỗ trợ mask bool hoặc uint8 nhị phân 0/1/255, cùng H×W với ROI.
- Kiểm tra shape, dtype, giá trị, diện tích mask và số pixel hợp lệ.
- Có policy `roi`, `empty`, `error` khi mask thiếu/rỗng/sai/không đủ diện tích.
- Ghi `mask_status`, `fallback_reason`, `mask_fraction`, `valid_pixels`, `total_pixels`.
- ROI rỗng hoặc ảnh không đúng contract bị báo lỗi rõ ràng.
- Không tự suy đoán foreground hoặc mask quần áo.

### 4.4. Similarity và validation

- Hellinger similarity: `1 - sqrt(max(0, 1 - sum(sqrt(p_i*q_i))))`.
- Thêm Histogram Intersection làm đối chứng ở giai đoạn triển khai ban đầu.
- Kiểm tra cùng version, chiều, config hợp lệ, finite, không âm và L1.
- Khi một histogram rỗng, similarity bằng 0, kể cả rỗng/rỗng.
- Không xem similarity là xác suất đúng sản phẩm.

## 5. FeatureBundle và contract đầu vào/đầu ra

File mới, sau đó được hoàn thiện: [AI/features.py](../AI/features.py).

### 5.1. Trích xuất kết hợp

- `extract_features()` trích embedding FashionCLIP và HSV cho cùng garment ID.
- Nhận riêng `embedding_image` và `roi_rgb`, giữ nguyên input encoder trong khi màu dùng ROI nguồn.
- `extract_detection_features()` là entry point nhận payload detection cho M3.

### 5.2. Adapter nguồn ROI

`color_input_from_detection()` ưu tiên:

1. `source_roi_base64`.
2. ROI cắt từ `original_rgb` bằng `original_bbox` hoặc `bbox`.
3. Crop cũ `image_base64` khi không có nguồn trên; ghi `legacy_processed_crop` và lý do fallback.

Ảnh/bbox được cung cấp nhưng sai sẽ báo lỗi, không âm thầm đổi nguồn. Mask base64 yêu cầu PNG; mask sai được đưa qua policy fallback có ghi nhận. Tọa độ mask phải là tọa độ ROI, không phải full-image mask.

### 5.3. Bundle thống nhất

- Dataclass `M3Features`, có alias công khai **`FeatureBundle`**.
- Schema version **`m3-features-v1`**.
- Chứa garment ID, embedding, embedding model/version, color descriptor và metadata.
- Có `.validate()`, `.to_dict()` và `.from_dict()` để kiểm tra và round-trip JSON.
- Kiểm tra schema version, identity, model version, metadata, mask status, pixel count và vector/descriptor.
- Không tạo vector hoặc bundle giả khi encoder lỗi.

### 5.4. Schema và profile bàn giao

Các file mới:

- [hsv147.json](m3_contract/hsv147.json): profile chung cho query và catalog.
- [m2_input.schema.json](m3_contract/m2_input.schema.json): schema payload adapter M3 nhận.
- [feature_bundle.schema.json](m3_contract/feature_bundle.schema.json): schema output M3 cho M4.

Profile hiện tại có color version **`hsv-hs-v1-147-b0681e3cbf842188`**. Hai JSON Schema đã được kiểm tra bằng `Draft202012Validator`; bundle thật và input PNG fixture validate thành công. Đây là contract bàn giao, chưa phải thay đổi schema HTTP production của M2.

## 6. FashionCLIP: giữ encoder và kiểm chứng weights thật

File sửa ở giai đoạn đầu: [AI/embedding_model.py](../AI/embedding_model.py).

- Giữ model ID `patrickjohncyh/fashion-clip`, tracking version `2`.
- Giữ CLIPModel, CLIPProcessor, preprocessing hiện có và L2 normalization.
- Bổ sung kiểm tra NaN/Inf, vector zero hoặc norm không hợp lệ; giữ kiểm tra 512 chiều.
- Không fine-tune, thay encoder hoặc đổi kích thước pgvector.

Ở audit giữa đợt, weights chưa có nên inference thật bị chặn. Đến giai đoạn handoff đã tải được snapshot gốc vào `.hf_cache/hub` và chạy thành công.

File mới: [AI/verify_m3.py](../AI/verify_m3.py).

- Smoke check độc lập với DB, YOLO và website; mặc định offline.
- Chạy encoder thật, kiểm tra FeatureBundle JSON round-trip và cache màu một ảnh.
- Xác minh query/catalog dùng cùng ảnh/config tạo descriptor bằng nhau.
- Lưu revision, hash weights, input/source hashes, môi trường và lỗi nếu có.

Kết quả đã chạy:

| Hạng mục | Kết quả |
|---|---|
| Weights revision | `7e3ba62ce16b379a1ab479346b66f192e76f51b7` |
| Thiết bị | CPU |
| Embedding dimension | 512 |
| Embedding L2 | `0.9999999714523827` |
| Color dimension | 147 |
| Color L1 | `1.0` |
| Bundle JSON round-trip | Pass |
| Query/catalog identical input | Descriptor bằng nhau |
| Mask trong smoke thật | Thiếu mask; fallback `mask_missing` được ghi rõ |

Ảnh thử là ảnh starter catalog, chưa phải ROI quần áo được M2 xác nhận. Không dùng kết quả smoke để kết luận chất lượng tìm kiếm.

Bằng chứng: [report.json](m3_handoff_evidence/report.json), [bundle.json](m3_handoff_evidence/bundle.json), [color_cache.json](m3_handoff_evidence/color_cache.json).

## 7. Cache màu catalog và tính tương thích query/catalog

File mới, sau đó được hoàn thiện: [AI/color_cache.py](../AI/color_cache.py).

- `build_color_cache()` tạo descriptor từ manifest local ROI/mask.
- Cache key gồm product ID, model/version, ROI policy, color version và hashes ảnh/mask.
- Cache màu tách khỏi embedding/catalog gốc; CLI từ chối ghi đè output đã tồn tại.
- Có coverage, danh sách failures và thống kê mask status.
- Ảnh thiếu hoặc không có pixel hợp lệ được ghi failure, không thay bằng descriptor giả.
- `read_cached_descriptor()` kiểm tra key integrity và identity/version/policy.
- `catalog_descriptor_for_query()` kiểm tra shared config, model/version, ROI policy, color version và product lookup.
- Thiếu ID báo `KeyError`; mismatch/ambiguous entry báo `ValueError`; M4 quyết định policy ranking.

Đã tạo cache mẫu **28/28 ảnh**, không failure; cả 28 thiếu mask nên fallback ROI. Đã đọc và validate lại toàn bộ cache trong lần audit. CLI batch, ảnh thiếu và chống overwrite đã có integration test.

Chưa có catalog production đầy đủ để chạy toàn bộ. Tài liệu handoff đã hướng dẫn manifest, lệnh batch, kiểm tra coverage/failures và chia shard nếu catalog lớn. CLI hiện làm việc với local files và JSON in-memory, không tự tải R2/HTTP hoặc tự detect ROI.

## 8. Tests và benchmark theo từng giai đoạn

### 8.1. Tests

File mới:

- [tests/test_hsv_descriptor.py](../tests/test_hsv_descriptor.py): **29 tests** cho histogram, mask, ROI adapter, embedding validation, cache và evaluator helpers.
- [tests/test_m3_handoff.py](../tests/test_m3_handoff.py): **7 tests** cho bundle round-trip, serialized contract lỗi, image payload, PNG mask, query/catalog mismatch và batch cache CLI.

Các ca quan trọng: áo trắng trên nền trắng, vùng da nhỏ, xám/đen, vải tối còn sắc, ít bão hòa, đa màu, mask sai/thiếu/rỗng, ROI lỗi, descriptor version mismatch, encoder lỗi và cache không ghi đè.

Tiến độ số lượng tests:

| Giai đoạn | Kết quả |
|---|---|
| Triển khai HSV ban đầu | 18 tests cũ + 29 tests mới = **47 pass** |
| Hoàn thiện handoff | 47 tests trước + 7 tests handoff = **54 pass, 0 fail, 0 error** |

Lệnh suite cuối:

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
git diff --check
```

Log cuối: [tests.txt](m3_handoff_evidence/tests.txt). Tests dùng mock/tensor fixture được phân biệt với smoke weights thật; không coi test mock là inference pretrained thật.

### 8.2. Benchmark giai đoạn triển khai ban đầu

File mới: [evaluation/run_color_benchmark.py](../evaluation/run_color_benchmark.py).

- Công cụ color-only offline, không cần DB.
- Có factorial ablation H-S/H-S+achromatic × ROI/mask, failures trong denominator, coverage, latency màu, per-query ranking, config và hashes.
- Có paired cluster bootstrap và manifest cho dữ liệu riêng.
- Giai đoạn đầu đã chạy fixture 8 catalog/24 query và 28 ảnh starter với nền được thêm có kiểm soát; khi đó có đối chứng baseline theo yêu cầu ban đầu.
- Giai đoạn audit/handoff sau **không chạy benchmark chất lượng mới hoặc lập thêm so sánh RGB**.

Kết quả lịch sử được giữ tại [m3_results/synthetic/report.md](m3_results/synthetic/report.md) và [m3_results/photos/report.md](m3_results/photos/report.md). Query photo được tạo từ chính catalog; mask chỉ loại nền thêm vào. Các kết quả này không chứng minh cải thiện retrieval độc lập hoặc chất lượng segmentation.

## 9. Tài liệu, cấu hình và môi trường

| File/thư mục | Thay đổi |
|---|---|
| `docs/M3_AUDIT.md` | Audit trước sửa source |
| `docs/M3_HSV_REPORT.md` | Báo cáo triển khai ban đầu, công thức, tests và giới hạn benchmark |
| `docs/M3_HANDOFF.md` | Schema, ví dụ input/output, hàm gọi, version, fallback, commands và trách nhiệm tích hợp |
| `docs/m3_contract/` | Profile config và JSON Schema bàn giao |
| `docs/m3_results/` | Cache mẫu, manifest, benchmark reports và log giai đoạn đầu |
| `docs/m3_handoff_evidence/` | Output và log kiểm chứng weights thật/handoff |
| `docs/M3_CHANGE_SUMMARY.md` | Bản tổng hợp toàn bộ thay đổi này |
| `requirements.txt` | Thêm NumPy và opencv-python-headless |
| `.gitignore` | Cho phép lưu artifact M3 cần thiết; weights/cache model vẫn được ignore |

Môi trường đã dùng cho kiểm chứng thật: Python 3.12.10, Torch 2.7.1+cpu, Transformers 4.57.1, NumPy 2.1.3, OpenCV 5.0.0 và Pillow 11.3.0. Đã bổ sung boto3/pgvector vào `.venv` để chạy tests cũ. Chưa xác minh lại trên đúng toàn bộ dependency pins của repository; đặc biệt Torch/Transformers môi trường thực khác pins. Weights local không commit.

## 10. Phần việc các bên làm tiếp

### M2 — Detection, ROI và mask

- Cung cấp source ROI chưa gamma/padding/resize và mask PNG binary cùng hệ tọa độ.
- Thống nhất bbox, ROI policy và provenance theo contract.
- Bổ sung payload HTTP phù hợp; giữ crop riêng cho encoder.
- Chạy integration thực detector/HTTP → M3, kiểm tra mask alignment và fallback.

### M4 — Retrieval, similarity và reranking

- Nhận FeatureBundle và nối M3 vào query pipeline.
- Chuẩn bị catalog đầy đủ, chạy cache màu, kiểm tra coverage/failures.
- Giữ query/catalog cùng config/version/ROI policy.
- Giữ FashionCLIP lấy candidates; lookup màu và dùng histogram similarity trong reranking.
- Quy định handling descriptor thiếu/mismatch và calibration trên validation.
- Tích hợp endpoint/frontend; kiểm thử retrieval, latency và API end-to-end.

### M3 — Hỗ trợ sau bàn giao

- Giải thích contract, cấu hình và kết quả validation.
- Sửa lỗi trong phạm vi extractor/bundle/cache nếu phát hiện khi tích hợp.
- Phối hợp xác minh lại bằng source ROI/mask và catalog thật khi được cung cấp.

## 11. Trạng thái hoàn thành

**Đã hoàn thành phần M3 có thể bàn giao độc lập:** HSV extractor, FeatureBundle, adapter, cache, contract, tests và pretrained FashionCLIP smoke check.

**Chưa hoàn thành pipeline production:** HTTP ROI/mask thật từ M2, toàn bộ catalog production, HSV reranking M4, API/frontend end-to-end và đánh giá retrieval độc lập. Không đánh đồng kết quả smoke hoặc cache mẫu với tích hợp production; không khẳng định chất lượng tìm kiếm đã tăng.
