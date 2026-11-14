# Hợp đồng M1 → M2 → M3, phiên bản m2.roi.v1

Ngày chốt triển khai M2: 07/11/2026; cập nhật phase 3 ngày 09/11. API đã có ROI nguồn, ID, metadata và mask PNG tùy chọn. Mode off mặc định trả null/mask_disabled; mode thử trả mask hoặc null/fallback theo gate. M1/M3 chưa cùng người phụ trách nghiệm thu; audit M2→M3 đã qua bằng code thực. Fixture phase 3: `tests/fixtures/m2/phase3_responses.json`.

**Lỗi review merge đã sửa trong phase 3:** M3 ưu tiên canonical PNG và vẫn đọc alias cũ. Gate `evaluation/check_m2_m3_contract.py` hiện báo `compatible`; evidence cũ needs_alignment được giữ tại thư mục review riêng. Báo cáo mới: `docs/m2/phase3/RESULTS.md`.

## M1 → M2 trong bộ nhớ

| Trường | Quy ước |
|---|---|
| `source_rgb` | NumPy uint8 H×W×3 RGB, ảnh sau chuẩn hóa hình học, trước gamma |
| `detection_rgb` | Cùng dtype/shape/hệ tọa độ source; có thể đã tăng sáng |
| `geometry_meta` | Kích thước upload/working, EXIF đã áp dụng, phép đổi tọa độ hoặc quy tắc transform đủ để ánh xạ nhãn |
| `quality_meta` | Mode tiền xử lý, gamma có bật hay không, cấu hình M1 |

M2 không sửa array nguồn tại chỗ. YOLO nhận `detection_rgb` chuyển RGB→BGR đúng một lần khi dùng NumPy; crop ROI lấy từ `source_rgb`. Khi chưa có M1, hai nhánh là cùng nội dung decode cũ với mode `legacy_identity`; không khẳng định đã xử lý đúng EXIF. Nếu shape khác nhau, báo lỗi đầu vào, không tự resize để che sai lệch.

## Bbox, ROI và ID

- `bbox_xyxy=[x1,y1,x2,y2]` là số nguyên trên **ảnh working của M1**, x2/y2 exclusive: `source_rgb[y1:y2,x1:x2]`. Khi chuyển float: floor x1/y1, ceil x2/y2; clip từng tọa độ vào [0,W]/[0,H]. Không đổi chỗ hai đầu box sai, không ép box rỗng thành 1 pixel.
- Box sau clip phải x2>x1 và y2>y1. Giữ class/confidence; mapping 10 lớp đã có phải khớp `model.names`.
- Phase 2 giữ policy crop legacy: từng variant chỉ được tạo khi cả hai cạnh crop nguồn ≥50. Không có variant hợp lệ thì không trả garment. Vì vậy crop thành công không đồng nghĩa detector recall; lưu số box/variant bị bỏ khi đo sau này.
- ROI màu dùng **tight bbox đã clip**, trước expand/pad/resize. Bbox của ROI màu được ghi trong `roi_metadata.bbox_xyxy`; `source_crop_width/height` cũ vẫn chỉ kích thước crop primary padded.
- Để giữ nguyên pixel/JPEG nhánh encoder, `original_bbox` và các legacy variants tiếp tục dùng quy tắc cũ `int()` trên bbox YOLO. `bbox`/`roi_metadata.bbox_xyxy` của ROI màu dùng floor/ceil và clip như trên, nên có thể rộng hơn một pixel ở cạnh phải/dưới. `roi_metadata.raw_bbox_xyxy` giữ float gốc; mỗi crop variant vẫn có bbox riêng. Không dùng kích thước crop padded để decode ROI màu. Đây là thay đổi hình học phase 2, phải tách khỏi tác dụng mask ở thí nghiệm sau.
- `query_id` do request/caller cung cấp hoặc service sinh UUID nếu vắng; giữ nguyên suốt response. Sắp garment theo `(class_id,x1,y1,x2,y2,-confidence)` rồi gắn `{query_id}_g000`, `_g001`… ID ổn định trong request với cùng detections, không cam kết qua request/model khác.

## M2 → M3 qua HTTP

Giữ nguyên endpoint `/detect/upload`, `success`, `total_crops`, `crops`, `message` và các trường cũ, nhất là `image_base64`/`crop_variants`. Không đổi số crop để làm thí nghiệm mask. Bổ sung trường tùy chọn:

| Cấp | Trường mới | Ý nghĩa |
|---|---|---|
| Response | `schema_version` | `m2.roi.v1` |
| Response | `query_id` | Định danh request |
| Response | `diagnostics` | Đếm raw box, bbox lỗi, prediction lỗi, box không có variant, số variant bị loại và số crop trả về |
| Crop | `query_id`, `garment_id` | Liên kết detection → màu/embedding → retrieval |
| Crop | `source_crop_png_base64` | Tight ROI nguồn, PNG 8-bit RGB, không alpha/padding |
| Crop | `mask_png_base64` | Null hoặc PNG grayscale 8-bit, Hroi×Wroi, chỉ 0/255, foreground=255 |
| Crop | `roi_metadata` | Hình học, trạng thái, version và lý do fallback |

PNG/JPEG đều là **raw base64**, không tiền tố `data:image/...`. OpenCV decode ROI thành BGR thì M3 chuyển sang RGB đúng một lần; Pillow dùng `convert('RGB')`. Mask decode grayscale, không convert RGB, không nội suy tuyến tính. Nếu cần resize mask ở consumer thì nearest-neighbor và ghi transform; ưu tiên giữ ROI và mask nguyên kích thước.

`roi_metadata` bắt buộc khi có ROI mới: `bbox_xyxy`, `working_width`, `working_height`, `roi_width`, `roi_height`, `coordinate_space='m1_working_image'`, `color_space='RGB'`, `roi_source`, `roi_status`, `fallback_reason`, `mask_method`, `mask_config_id`, `geometry_mode`. Thông tin transform M1 đi kèm khi có; nếu chưa có thì ghi mode thiếu, evaluator không đo GT khác hệ tọa độ.

Phase 3 bổ sung `segmentation` chứa config/version, flag provisional, threshold/polarity, contrast/area/change, latency và error_type khi có. Raw/processed masks chỉ giữ trong bộ nhớ hoặc artifact offline, không nhét array vào JSON. M3 giữ nguyên metadata này dưới `color.metadata.m2_roi_metadata`, cùng query/garment ID; không để dữ liệu M2 ghi đè `roi_policy`/config HSV. Null canonical mask không được thay bằng mask alias cũ; source canonical cung cấp mà lỗi không âm thầm fallback JPEG.

| Trạng thái | Mask | Consumer màu |
|---|---|---|
| `masked` | PNG hợp lệ | Chỉ pixel foreground của ROI nguồn |
| `unmasked` | Null, lý do `mask_disabled` | Toàn ROI nguồn; control H0 sau merge (R0 là tên lịch sử) |
| `fallback` | Null, lý do cụ thể | Toàn ROI nguồn và ghi nhận fallback |
| Không có trường mới | Không có | Policy legacy công bố rõ, không giả vờ đã chạy mask |

Lý do dự kiến: `roi_too_small`, `low_contrast`, `ambiguous_foreground`, `empty_mask`, `area_out_of_range`, `morphology_unstable`, `mask_processing_error`. Danh sách và ngưỡng cụ thể sẽ khóa ở phase 3; diện tích hợp lý không chứng minh mask đúng ngữ nghĩa. Không trả mask rỗng giả làm hợp lệ. Mask lỗi consumer phải chuyển fallback có ghi lý do; không dùng mask sai shape hoặc bỏ qua lỗi một cách im lặng.

Mask chỉ dùng nhánh màu. Encoder tiếp tục nhận crop JPEG legacy. M3 chịu trách nhiệm decoder và descriptor dùng chung cho query/catalog; M2 không sửa histogram/ranking ở phase 1.

## Hiện trạng và phần việc tiếp theo

- Phase 1: model path/Docker, fixture legacy/candidate, kiểm thử bootstrap và baseline regression.
- Phase 2: `detector.py` thêm đường vào hai nhánh, clip/crop nguồn/ID; `schemas.py` khai báo trường mới; `utils.py` encode PNG; `api.py` nối PreparedImage.
- Phase 3: `segmentation.py`, Otsu/morphology/gate và metadata.
- M1 cung cấp PreparedImage/transform; M3 tiêu thụ fixture và cùng test round-trip. Chưa gửi thông tin ra ngoài hoặc thay code của M1/M3 trong phase 1.

## Fixture nghiệm thu

1. Hai nhánh cùng shape nhưng pixel khác nhau; màu áo nguồn RGB (30,100,200), nhánh sáng (70,140,240).
2. ROI 64×64 đúng lát cắt `[16:80,32:96]`; mask tham chiếu tổng hợp khớp vùng áo.
3. Base64 PNG round-trip không đổi pixel; ba JPEG legacy vẫn 224×224.
4. Consumer schema cũ nhận cả legacy và candidate payload; trường mới bị bỏ qua ở schema cũ, chưa phải hỗ trợ decoder mới.
5. Có bbox chạm biên/ngoài ảnh/nhỏ/đảo/rỗng; kiểm thử clip production bổ sung ở phase 2.

## Cách M1/M3 nối vào sau phase 2

```python
# Các array đã được M1 xử lý, cùng hình học và dtype uint8 RGB.
batch = detector.detect_prepared(
    source_rgb, detection_rgb,
    query_id="query_001",  # Bỏ qua để service sinh UUID.
    geometry_meta={"mode": "m1_normalized", "working_width": source_rgb.shape[1],
                   "working_height": source_rgb.shape[0]},
    quality_meta={"gamma_enabled": True},
)
response = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics)
```

M1 bổ sung transform/EXIF đầy đủ vào geometry metadata thật; ví dụ trên chỉ minh họa lời gọi, không thay thế transform. `detect_prepared` trả `DetectionBatch`, không phải list. `detect_from_bytes` và `_process(BGR)` vẫn trả list cho caller cũ; API dùng `detect_batch_from_bytes` để giữ metadata kể cả khi không có detection.

Upload hỗ trợ `POST /detect/upload?query_id=query_001`. ID tối đa 128 ký tự, bắt đầu bằng chữ/số ASCII, phần còn lại có thể chứa `_.:-`; ID sai trả 422. Bytes không decode được trả 400. Không có detection vẫn trả HTTP 200, `success=false`, crop rỗng như cũ, cộng query/diagnostics.

Consumer đã nối: M3 decode `source_crop_png_base64` thành RGB và dùng mask hợp lệ đúng ROI; mask null dùng toàn ROI theo policy HSV đang có. Alias cũ giữ tương thích. Schema input M3 đã cập nhật trong `docs/m3_contract/m2_input.schema.json`. Phase 1 candidate masked vẫn chỉ là reference tổng hợp; fixture phase 3 chứa mask Otsu tính thật. Không có mask GT thật hoặc calibration validation trong phase 3; xem báo cáo kết quả.
