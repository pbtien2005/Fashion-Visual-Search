# Tiến độ M2

Kế hoạch chuẩn: [M2_PLAN.md](M2_PLAN.md). Cập nhật ngày 07/11/2026.

## Trạng thái

| Phase | Trạng thái | Bằng chứng |
|---|---|---|
| 1 — Môi trường/hợp đồng | Hoàn thành | Docker/model thật, contract, fixture, manifest, 24 test qua, HTTP smoke, baseline 16 ảnh; xem `M2_ENVIRONMENT.md` |
| 2 — ROI/HTTP | Hoàn thành | Hai nhánh RGB, clip/ROI/ID/PNG HTTP; 40 test qua; 54 ROI và 161 JPEG đã kiểm trên ảnh thật; xem `docs/m2/phase2/RESULTS.md` |
| 3 — Mask/morphology | Chưa bắt đầu | Chưa có segmentation |
| 4 — Dữ liệu/evaluator | Chưa bắt đầu | Nhãn cũ chưa có target bbox/mask |
| 5 — Thực nghiệm/tích hợp | Chưa bắt đầu | Chưa chạy baseline hoặc đối chứng mới |
| 6 — Báo cáo/bàn giao | Chưa bắt đầu | Mới có kế hoạch |

## Đầu vào còn thiếu hoặc chưa xác minh

- Model `object_detection/best.pt` đã load bằng Ultralytics 8.4.31, task detect, mapping đủ 10 lớp khớp project; hash trước/sau không đổi. Docker đã mount file read-only; health và upload thật hoạt động.
- Chưa cấu hình CUDA; baseline chạy CPU Docker. DB/encoder/main API/frontend không bật trong phase 1.
- Hợp đồng M2 đã lưu tại `M2_CONTRACT.md`, chưa được M1/M3 cùng nghiệm thu. M2 đã có `detect_prepared` và PNG response; upload dùng `legacy_identity` khi chưa có M1. M3 decoder/descriptor chưa sửa; mask production hiện null.
- Cần 20 ảnh phần Người 2 (5 validation + 15 test), 5 mask tham chiếu (2 validation + 3 test) và kiểm chéo; manifest nháp đang rỗng. 16 ảnh outfit cũ chỉ là regression.

## Lượt lập kế hoạch

- Đọc đủ 17 trang văn bản bản bàn giao; đối chiếu module, schema, serializer, HTTP client, Docker context, runner/nhãn hiện có.
- Xác nhận 54 ảnh hiện có: 10 smoke + 28 catalog + 16 outfit.
- Tạo `M2_PLAN.md`, file tiến độ này và hướng dẫn nhận lệnh phase trong `AGENTS.md`.
- Không sửa code sản phẩm, không chạy test, không khẳng định runtime hoặc cải thiện metric.

## Phase 1 — Đã hoàn tất ngày 07/11/2026

- Lượt local bị gián đoạn đã chuyển sang Docker theo yêu cầu người dùng. Đã xóa `.venv`, `tmp/python`, `tmp/python-bin`, `tmp/uv-cache`; giữ bản trích PDF cũ để tra cứu. Không xóa weights hoặc thay đổi của người dùng.
- Sửa tối thiểu: `object_detection/config.py` chọn explicit path → environment → mount/local/legacy path; `model_available` chỉ chấp nhận file. API dùng chung resolver. Compose mount `./object_detection/best.pt` read-only vào đường dẫn model cũ.
- Dockerfile và `requirements-m2.txt` dựng Python 3.12.12, torch 2.11 CPU, Ultralytics 8.4.31; `.dockerignore` loại weights. Không train hoặc đổi model.
- Đã tạo hợp đồng `M2_CONTRACT.md`, hướng dẫn `M2_ENVIRONMENT.md`, fixture tổng hợp + legacy/candidate response dưới `tests/fixtures/m2/`, manifest regression có hash/provenance và manifest gán nhãn nháp dưới `evaluation/m2/`.
- Kiểm thử: 8 model-path + 3 API bootstrap + 6 fixture + 4 color cũ + 3 benchmark cũ = **24 test qua**. Test path đã fail trước khi thêm resolver; fixture test đã fail khi thiếu artifact rồi pass sau khi sinh. Coverage detection **88%**, helper mới `config.py` **100%**, kết hợp unit/fixture và đường chạy YOLO thật. Compile và `pip check` qua.
- Chưa chạy hai nhóm search/vectorizer cần môi trường API/encoder ngoài phạm vi image detection; không gọi 24 test này là toàn bộ suite.
- Service `fashion_object_detection` đang chạy tại `localhost:8001`; health `model_available=true`, `model_loaded=true`. Real HTTP upload ảnh outfit trả 200, 2 crop; JPEG 224×224; MIME text trả 400. Bằng chứng `docs/m2/phase1/http_smoke.json`.
- Baseline cuối không coverage: **16/16 ảnh xử lý không lỗi, 15/16 có crop, tổng 54 crop**; mean M2 948 ms, p95 1493 ms, CPU limit 4/RAM 4 GiB, một warm-up. Chỉ là decode/detection/crop/serialize, không gồm mạng/embedding/DB. Chưa có nhãn bbox/mask nên **không tính recall/mAP/IoU**. Không dùng tỷ lệ có crop 93,75% như accuracy.
- Artifacts: `docs/m2/phase1/baseline_20261107_cpu/{baseline.json,baseline.csv,detections.jpg}`, `coverage.txt`, `requirements.lock.txt`, `source_snapshot.json`. Lượt instrumented trung gian đã dọn sau khi lưu coverage.
- Đã xem contact sheet fixture và bbox thật; có ca không phát hiện `real_red_dress_01`, ca shorts bị gọi skirt và ảnh nhiều người có nhiều box. Đây là quan sát chẩn đoán, cần nhãn ở phase 4 để định lượng đúng target.
- Phần candidate ROI/mask mới chỉ là fixture và đặc tả; API production chưa trả trường này. Không đánh dấu phase 2/3 đã làm.
- Bước chuyển sau phase 1 là phase 2; phần này đã hoàn thành bên dưới.

## Phase 2 — Đã hoàn tất ngày 07/11/2026

- Thêm `object_detection/roi.py` và `DetectionBatch`; `detect_prepared` nhận source/detection uint8 RGB, kiểm tra hình học và metadata; `detect_batch_from_bytes` giữ query/counters kể cả không có crop. API hiện dùng adapter identity vì M1 thật chưa có.
- YOLO chỉ dùng nhánh detection; ROI tight PNG và legacy JPEG dùng source. Bbox ROI floor/ceil và clip; `original_bbox`/JPEG giữ int legacy. Source ROI là bản copy, không sửa ảnh đầu vào. ID được sắp thứ tự xác định trong query.
- Mở rộng schema/serializer/API bằng các trường tùy chọn: version, ID, ROI PNG, metadata, diagnostics; mask null với `mask_disabled`. Invalid image 400, invalid query ID 422. Không đổi endpoint hoặc trường JPEG/crop variants cũ.
- Test-first: 16 test phase 2 đã fail trước triển khai rồi qua sau sửa. Tổng **40 test qua** (27 detection/API/model-path + 6 contract fixture + 7 color/benchmark cũ); coverage detection từ tests **89%**, `roi.py` 98%. Không tuyên bố đã chạy hai nhóm search/vectorizer cần môi trường API ngoài image detection.
- Model thật trên 16 ảnh regression: **54 crop giữ nguyên số lượng**, **161 JPEG variants khớp cả pixel và base64** với source phase 1 đóng băng; **54 ROI PNG khớp chính xác source slice**. Weights không đổi. 53 bbox ROI làm tròn ra ngoài khác bbox int cũ, được ghi riêng; không làm đổi JPEG.
- HTTP service đã rebuild và chạy thật: upload 200, 2 PNG ROI đúng pixel, ID đúng, mask null. Schema cũ thực từ image phase 1 đọc response mới được. Main API/frontend/DB/ranking chưa đổi.
- Payload cộng dồn 16 ảnh: 5.704.986 → 19.538.625 byte (3,425×) vì thêm PNG ROI nguồn; không coi chi phí truyền là không đáng kể. Chưa đo retrieval hoặc segmentation accuracy.
- Artifacts: `tests/fixtures/m2/phase2_response.json`; `docs/m2/phase2/{RESULTS.md,regression.json,http_smoke.json,coverage.txt,source_snapshot.json}`. Hợp đồng và README fixture đã cập nhật; tag `fashion-m2:phase1` giữ nguyên để làm đối chứng.
- Không có nhãn bbox/mask thật mới; manifest 20 ảnh vẫn pending theo phase 4. Chưa triển khai Otsu/morphology.
- **Tiếp theo: `làm phase 3`** — mask Otsu có điều kiện, morphology nhẹ, PNG mask/fallback theo contract.

## Mẫu cập nhật sau mỗi phase

- Thời điểm và phase:
- Việc đã hoàn thành, file đã sửa:
- Lệnh kiểm thử/đánh giá và kết quả thực tế:
- Artifact/config/snapshot:
- Thời gian đã dùng nếu biết:
- Đầu vào đang chờ và phần tiếp tục độc lập:
- Quyết định/phạm vi thay đổi:
- Điều kiện còn thiếu trước khi hoàn thành phase:
- Phase tiếp theo:
