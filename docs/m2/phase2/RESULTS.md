# Phase 2 — ROI nguồn và HTTP

Hoàn thành ngày 07/11/2026. Contract hiện tại: `docs/M2_CONTRACT.md`. Phase 3 chưa bắt đầu; mask production vẫn null.

## Thay đổi đã triển khai

- `ClothingDetector.detect_prepared(source_rgb, detection_rgb, ...)` nhận hai array uint8 RGB cùng hình học. YOLO dùng nhánh detection chuyển BGR; ROI và JPEG lấy từ source. Kiểm tra shape/dtype/metadata trước inference; không sửa array M1 tại chỗ.
- `DetectionBatch` giữ query ID và counters kể cả response rỗng; adapter bytes dùng `legacy_identity` tới khi có M1 thật. `detect_from_bytes` và `_process(BGR)` giữ kiểu trả list cho caller cũ.
- Bbox ROI màu làm tròn ra ngoài, clip vào ảnh, x2/y2 exclusive; bỏ bbox rỗng/đảo/ngoài ảnh/NaN. Giữ `original_bbox` và cách `int()` cũ cho nhánh JPEG để không đổi crop encoder. Metadata ghi cả float gốc và bbox từng nhánh.
- Gắn ID theo thứ tự `(class_id,x1,y1,x2,y2,-confidence)` trong request. API có `query_id` tùy chọn; vắng thì UUID. Giữ JPEG/variants; thêm ROI nguồn PNG, metadata, version và diagnostics. Mask null, `roi_status=unmasked`, `fallback_reason=mask_disabled`.
- Bytes không decode được trả HTTP 400, ID sai trả 422; hành vi không detection giữ HTTP 200 + `success=false` như cũ, cộng metadata request.

## Bằng chứng

| Kiểm chứng | Kết quả |
|---|---|
| Test detection/ROI/API/contract | 33 test qua |
| Test màu và benchmark cũ liên quan | 7 test qua |
| Tổng test đã chạy | **40 qua** |
| Coverage detection từ unit/integration test | **89%**; `roi.py` 98%, schema/config 100% |
| Bộ ảnh regression thật | 16 ảnh, 54 garment crop, cùng số lượng phase 1 |
| JPEG variant | **161/161 khớp từng pixel và base64 JPEG** với code phase 1 đóng băng |
| ROI PNG | **54/54 decode khớp chính xác lát cắt ảnh nguồn** |
| ROI bbox mới | 53/54 bbox có thay đổi làm tròn ra ngoài; JPEG bbox/pixel giữ nguyên |
| Ngưỡng crop legacy | Giữ cạnh tối thiểu 50; có 1 variant bị bỏ như cũ |
| HTTP thật | Upload 200, 2 ROI PNG đúng pixel, query/garment ID đúng, mask null |
| Consumer cũ | Schema thực trong image phase 1 đọc được response HTTP mới |
| Model | SHA-256 không đổi |

Không có bbox/mask target người kiểm tra, nên không tính recall/mAP/IoU hoặc cải thiện retrieval. M1 thật và M3 decoder chưa tích hợp; hai nhánh đã được kiểm bằng fixture có pixel khác nhau. Đây là giới hạn kiểm chứng, không chặn việc bàn giao interface phase 2.

Payload trên toàn 16 ảnh (JSON compact, raw base64): phase 1 **5.704.986 byte**, phase 2 **19.538.625 byte**, khoảng **3,425 lần**. Chênh lệch chủ yếu do thêm PNG ROI nguồn độ phân giải gốc. Đây là chi phí đã đo; phase 2 giữ PNG theo hợp đồng để bảo toàn pixel màu, chưa tối ưu nén/resize.

Artifacts: `regression.json`, `http_smoke.json`, `coverage.txt`, `source_snapshot.json` trong thư mục này. Fixture bàn giao: `tests/fixtures/m2/phase2_response.json`, dùng `source_rgb.png` và `detection_rgb.png` cạnh nó.

## Lệnh chạy lại từ root PowerShell

Giữ nguyên image `fashion-m2:phase1` để so source mới với code cũ trong `/app`. Không build đè tag này bằng source phase 2. Service Compose dùng image riêng và có thể build lại bình thường.

```powershell
$m2Root = (Get-Location).Path
$m2Bind = "type=bind,source=$m2Root,target=/workspace"

# Test không cần weights hoặc network.
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m coverage run --source=object_detection -m unittest discover -s tests -p 'test_detection*.py' -v
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m coverage run --append --source=object_detection -m unittest discover -s tests -p test_m2_contract_fixtures.py -v
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m coverage report -m

# Sinh output thật của serializer với detector giả lập một bbox.
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m evaluation.check_m2_phase2 fixture

# Kiểm chứng model thật; dùng thư mục mới để giữ report đã lưu.
docker run --rm --network none --cpus 4 --memory 4g --mount $m2Bind -w /workspace -e YOLO_CONFIG_DIR=/tmp -e YOLO_AUTOINSTALL=false --entrypoint python fashion-m2:phase1 -m evaluation.check_m2_phase2 regression --output output/m2/runs/phase2_repeat_01

docker compose up -d --no-deps --build object-detection
docker run --rm --network container:fashion_object_detection --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m evaluation.check_m2_phase2 http --output output/m2/runs/phase2_http_repeat_01
```

Lệnh regression cố ý cần image phase 1 còn nguyên source cũ; script báo lỗi nếu `/app/detector.py` trùng source mới. Không coi build từ source mới là tái tạo code baseline cũ. Các hash nguồn/image đã lưu để kiểm tra; chưa đóng gói image đối chứng cho máy khác trong phase này.

## Bàn giao và bước kế tiếp

M1 gọi `detect_prepared` bằng hai nhánh cùng hình học và transform thật. M3 nhận `source_crop_png_base64` + `roi_metadata`; khi mask null, dùng toàn ROI nguồn và ghi chế độ unmasked. Không lấy padding hoặc ảnh detection đã tăng sáng để tính màu. `source_crop_width/height` legacy thuộc crop padded, khác `roi_width/height` của ROI tight.

Phase 3 sẽ thêm Otsu, morphology, mask PNG nhị phân và fallback có điều kiện. Không thay nhánh JPEG hoặc ranking khi đo tác động mask. Các bbox target phải dùng đúng quy tắc working/rounding đã công bố.
