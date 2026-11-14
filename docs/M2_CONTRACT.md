# Hợp đồng M1 → M2 → M3: m2.roi.v1

Endpoint giữ `POST /detect/upload`, multipart field `file`. Legacy `success`, `total_crops`, `crops`, `message`, `image_base64`, `crop_variants` vẫn giữ. Cấu hình hiện tại: M1 identity, mask off; xem [bàn giao M4](M4_HANDOFF.md).

## Ảnh và tọa độ

Upload gọi M1 `prepare_image` một lần rồi M2 `detect_prepared`. `source_rgb`/`detection_rgb` cùng H×W×3 uint8 RGB; source đã EXIF/RGB/alpha/resize nhưng chưa gamma, detection có thể gamma. M2 chuyển RGB→BGR một lần cho YOLO; crop màu từ source, không sửa array nguồn tại chỗ. Khác shape là lỗi.

`bbox_xyxy=[x1,y1,x2,y2]` trên working canvas M1, x2/y2 exclusive: `source_rgb[y1:y2,x1:x2]`. Floor đầu, ceil cuối, clip vào [0,W]/[0,H]; loại box rỗng/đảo. Geometry ghi working/oriented size, scale và EXIF. Không áp bbox working vào upload chưa EXIF/resize.

ROI màu là tight bbox, không expand/pad/resize. `original_bbox`/JPEG variants giữ `int()` legacy, có thể khác một pixel với floor/ceil ROI màu. `source_crop_width/height` legacy là kích thước primary padded; kích thước PNG lấy từ decode hoặc `roi_metadata`.

Legacy variant chỉ tạo khi hai cạnh crop nguồn ≥50; box không có variant hợp lệ bị bỏ, diagnostics ghi nhận. Số crop không phải số target đúng.

## Payload

| Cấp | Trường | Ý nghĩa |
|---|---|---|
| Response | `schema_version` | `m2.roi.v1` |
| Response | `query_id` | Caller ID hoặc UUID service sinh |
| Response | `diagnostics` | Raw/invalid boxes, prediction lỗi, variant bị loại, crop trả về |
| Crop | `query_id`, `garment_id` | Nối detection → feature → retrieval |
| Crop | `source_crop_png_base64` | Tight source ROI PNG uint8 RGB |
| Crop | `mask_png_base64` | Null hoặc PNG grayscale uint8 0/255, foreground=255 |
| Crop | `roi_metadata` | Geometry, mask status, config/version, lý do fallback |

Payload ảnh là raw base64, không có tiền tố data URL. PNG nguồn lossless; OpenCV decode phải BGR→RGB một lần, Pillow dùng RGB. Mask cùng W/H với ROI, không nội suy tuyến tính hoặc dùng shape padded để decode.

Metadata: `bbox_xyxy`, `working_width/height`, `roi_width/height`, `coordinate_space=m1_working_image`, `color_space=RGB`, `roi_source`, `roi_status`, `fallback_reason`, `mask_method`, `mask_config_id`, `geometry_mode`, transform M1 nếu có. `segmentation` giữ config/provisional/threshold/polarity/contrast/area/change/latency/error khi có; không đưa array vào JSON.

Garment sort `(class_id,x1,y1,x2,y2,-confidence)` rồi gắn `{query_id}_g000`…; không phải tracking ID qua request. Query ID dài 1–128, bắt đầu chữ/số ASCII, phần còn lại chữ/số/`_.:-`.

## Mask và fallback

| `roi_status` | Mask | Màu |
|---|---|---|
| `masked` | PNG hợp lệ | Foreground của source ROI |
| `unmasked` | Null, `mask_disabled` | Toàn source ROI |
| `fallback` | Null, lý do cụ thể | Toàn source ROI, lưu lý do |
| Legacy thiếu trường mới | Không có | Policy legacy công bố rõ, không coi đã phân vùng |

Mode thử: `raw`, `opening`, `closing`, `open_close`; Otsu với heuristic center/border/contrast/area, morphology ellipse 3×3 một lần. Không phải semantic segmentation. Gate có thể trả `roi_too_small`, `low_contrast`, `ambiguous_foreground`, `empty_mask`, `area_out_of_range`, `morphology_unstable`, `mask_processing_error`; không trả mask rỗng như thành công.

M3 ưu tiên canonical PNG/mask, vẫn hỗ trợ alias `source_roi_base64`/`mask_base64`. Canonical null mask thắng alias. Source canonical hỏng không âm thầm dùng JPEG; mask lỗi/shape sai theo policy fallback/error có ghi nhận. Metadata dưới `color.metadata.m2_roi_metadata`, không ghi đè identity HSV.

Mask chỉ tác động nhánh màu. Encoder JPEG giữ đường cũ. `detect_from_bytes`/`_process(BGR)` legacy trả list; API dùng `detect_batch_from_bytes` giữ query/diagnostics kể cả rỗng. `_process` legacy không tự có đầy đủ EXIF/transform M1.

## HTTP và kiểm chứng

Query ID sai →422, bytes không decode →400, thiếu model →503. Không detection →HTTP 200, `success=false`, crop rỗng. API chính có thể chuyển thành tìm kiếm rỗng theo hành vi hiện hữu.

Schema: [m2_input.schema.json](m3_contract/m2_input.schema.json). Fixture ở `tests/fixtures/m2/`: hai nhánh, PNG round-trip, box chạm biên/ngoài/rỗng/nhỏ, legacy response và Otsu response. Xem [EVALUATION](EVALUATION.md) để chạy; synthetic không thay GT thật. Kết quả và giới hạn tại [M4_HANDOFF](M4_HANDOFF.md).
