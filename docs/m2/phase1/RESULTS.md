# Kết quả phase 1 — 07/11/2026

Phase 1 đã hoàn thành. Hướng dẫn và lệnh tái lập: `docs/M2_ENVIRONMENT.md`. Contract: `docs/M2_CONTRACT.md`.

| Kiểm chứng | Kết quả |
|---|---|
| Model thật | Load được; task detect; 10 lớp khớp mapping |
| Tính toàn vẹn weights | SHA-256 trước/sau khớp; model mount read-only |
| Kiểm thử | 17 test M2 mới + 7 test cũ liên quan qua |
| Coverage detection | 88%; config resolver mới 100% |
| Dependency và cú pháp | pip check + compile qua |
| HTTP thật | Health OK; upload outfit 200, 2 crop; JPEG 224×224; MIME text 400 |
| Regression baseline | 16 ảnh, 0 lỗi xử lý; 15 ảnh có crop; tổng 54 crop |
| Latency CPU không instrumentation | Mean 948 ms; median 1011 ms; p95 1493 ms |

Baseline dùng `conf=0.5`, `imgsz=640`, NMS IoU 0.7, rect=true, max_det=300, crop expand 0/0.15/0.30 và ngưỡng cạnh crop cũ 50 pixel. Docker giới hạn CPU 4, RAM 4 GiB; torch 8 thread, một blank warm-up. Latency gồm decode/YOLO/crop/serialization; không gồm HTTP, M1, embedding, DB hoặc retrieval. Cỡ mẫu nhỏ và warm-up chưa phủ mọi shape; không coi là benchmark GPU.

`baseline_20261107_cpu/baseline.json` có model/config/manifest hash, metadata train lịch sử và từng prediction; CSV có timing/payload từng ảnh. `detections.jpg` là contact sheet đối chiếu ảnh/bbox. Metadata train được đọc từ checkpoint, không phải chạy lại tập validation cũ.

Tỷ lệ có crop **93,75% không phải accuracy/recall/mAP**. Chưa có bbox target/mask người kiểm tra nên không tính được các metric này. 16 ảnh là bộ cũ từng dùng phát triển, không phải test độc lập.

Quan sát từ hình: `real_red_dress_01` không trả crop; `real_shorts_02` trả class skirt; ảnh nhiều người có nhiều box, cần ghép đúng target ở phase 4. Không dùng crop của người/món khác để tính trúng.

Fixture candidate PNG ROI/mask đã sẵn cho M1/M3 nhưng trường mới chưa triển khai trong production. Manifest nháp 20 ảnh mới và 5 mask còn trống; dữ liệu/nhãn thật thuộc phase 4. Phase 2 bắt đầu từ source ROI và HTTP theo contract, không cần dựng lại môi trường.
