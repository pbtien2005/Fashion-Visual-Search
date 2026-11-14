# Phase 3 — Otsu, morphology và nối M3 — 09/11/2026

Đã hoàn thành phần triển khai/kiểm thử. **Cấu hình mới là khởi điểm, chưa được chọn bằng validation thật; service thường giữ `M2_MASK_MODE=off`.** Nhãn/mask thật và quyết định bật cấu hình thuộc phase 4–5.

## Kết quả đã kiểm chứng

| Hạng mục | Bằng chứng |
|---|---|
| Hợp đồng M2→M3 | Audit chuyển từ needs_alignment sang compatible; ROI 64×64 đúng pixel dù legacy bbox 63×63; canonical mask dùng 2.816 pixel foreground |
| Alias cũ | `source_roi_base64`/`mask_base64` vẫn đọc được |
| Chế độ | off, raw, opening, closing, open_close; kernel ellipse 3×3, mỗi phép một vòng |
| Kiểm thử toàn repo | 132 ca chạy: **131 qua, 1 bỏ qua** (pretrained opt-in); native exit code 0 |
| Coverage được đo | 94% trên object_detection + AI.features + AI.hsv_descriptor; segmentation.py 98%, features.py 96% |
| Regression model thật | 16 ảnh cũ, 54 ROI; cả 4 chế độ đưa đúng số pixel vào HSV |
| HTTP thật khi raw bật | HTTP 200, 1 crop có mask + 1 fallback; PNG đúng shape/0–255, HSV dùng đúng foreground |
| Model/HSV | Không đổi weights, histogram profile hoặc ranking weights; M3 runtime ưu tiên garment_id |

Unit/integration tests dùng encoder/DB fixtures như các test sẵn có. Ca pretrained được bỏ qua vì lượt này không tải model FashionCLIP. Smoke HTTP dùng YOLO thật và decoder/HSV thật, **không có encoder/DB/retrieval thật**. Không gọi đây là đánh giá end-to-end chất lượng tìm kiếm.

## Phương pháp và giới hạn

`object_detection/segmentation.py` nhận ROI uint8 RGB, không sửa nguồn. Chuyển gray, đo tương phản p95−p05, dùng Otsu chia hai miền. So cả hai polarity sáng/tối, chọn miền có tỷ lệ ở trung tâm cao hơn ở viền. Vùng trung tâm là 50% mỗi chiều; viền có bề dày 5% cạnh ngắn, tối thiểu 1 pixel. Đây là giả định đối tượng nằm gần tâm bbox, không phải nhận dạng ngữ nghĩa áo/da/nền.

Khởi điểm: cạnh ROI ≥8; tương phản ≥15 mức gray; foreground chiếm 3–95% ROI; tỷ lệ foreground vùng tâm ≥0,55 và chênh tâm−viền ≥0,10; thay đổi sau morphology ≤15% toàn ROI. Không quét nhiều kernel, không chọn component lớn nhất, không fill toàn bộ lỗ. Kernel cố định ellipse 3×3. Tham số/version đầy đủ nằm trong `mask_config_candidate.json` và metadata từng kết quả; không gọi các giá trị này là tối ưu.

Fallback trả mask null và lý do: `roi_too_small`, `low_contrast`, `ambiguous_foreground`, `empty_mask`, `area_out_of_range`, `morphology_unstable`, `mask_processing_error`. M3 dùng toàn ROI, ghi `mask_status=missing`; lý do M2 vẫn nằm trong `color.metadata.m2_roi_metadata`. Nếu PNG mask hỏng/sai shape/không nhị phân ở consumer, policy HSV hiện có quyết định roi/empty/error; không âm thầm đổi sang JPEG.

Hình `real_contact.jpg` có ca được chấp nhận vẫn lẫn mặt/tay/nền hoặc làm mất một phần vải. Do đó diện tích/tâm/viền hợp lý **không chứng minh mask đúng**. Không thay tham số dựa trên tập regression rồi coi tập đó là test độc lập.

## Regression smoke, không có ground truth thật

| Chế độ | Có mask | Fallback | Tổng ROI |
|---|---:|---:|---:|
| raw | 40 | 14 | 54 |
| opening | 40 | 14 | 54 |
| closing | 39 | 15 | 54 |
| open_close | 39 | 15 | 54 |

Raw fallback gồm 13 ambiguous_foreground và 1 area_out_of_range. Open_close gồm 13 ambiguous_foreground, 1 morphology_unstable, 1 area_out_of_range. Đây là tỷ lệ chấp nhận/fallback, **không phải accuracy hay IoU**. Không chọn mode bằng số mask được chấp nhận.

`regression.json` lưu từng ROI/mode, config, threshold/polarity, diện tích, thay đổi, latency helper và pixel count HSV. Cả 4 mode dùng cùng ROI; không đổi YOLO/embedding/ranking. Latency chỉ của segmentation, không gồm inference/encode/decode/mạng/HSV.

## Bằng chứng hình và fixture

- `synthetic_contact.png`: ROI, mask raw, sau open_close, reference GT tổng hợp và overlay. Có sáng/tối, noise/lỗ, trắng trên trắng và phân chia mơ hồ.
- `synthetic_masks/`: các mask raw/processed/reference tương ứng. `synthetic.json` lưu IoU tổng hợp với fallback tính 0; chỉ là functional QA, không phải validation quần áo thật.
- `real_contact.jpg`: ba ROI raw được chấp nhận đầu tiên và ba fallback đầu tiên, không có GT; nguồn/tác giả/license kế thừa `evaluation/m2/regression_manifest.json`.
- `tests/fixtures/m2/phase3_responses.json`: output thật của detector/serializer trên fixture YOLO giả lập cho raw/morph/fallback/off. Khác với masked example phase 1, mask ở đây thực sự được tính bằng Otsu.
- `contract_audit.json`, `http_enabled_mask.json`, `test_summary.json`, `tests.txt`, `coverage.txt`, `requirements.test.lock.txt`, `source_snapshot.json`: audit và môi trường.

## Code và cấu hình

- `AI/features.py`: ưu tiên canonical PNG, giữ alias; null mask canonical thắng alias cũ; source hỏng không fallback JPEG; metadata/ID được giữ; không cắt working bbox lên original khác hình học.
- `api/search_service.py`: hai điểm gọi ưu tiên garment_id; không đổi công thức/weights ranking.
- `object_detection/detector.py`: nhận `MaskConfig`, chạy helper trên source ROI và giữ raw/processed mask trong bộ nhớ; JPEG encoder giữ nguyên.
- `object_detection/utils.py`: encode mask PNG grayscale nhị phân 0/255, từ chối shape/dtype/giá trị sai.
- `RoiMetadata.segmentation`: config, provisional flag, threshold/polarity/area/change/latency. Không thêm array raw vào JSON.

`MaskConfig()` và Compose mặc định off. `M2_MASK_MODE=raw|opening|closing|open_close` bật thử ở service. `M2_MASK_CONFIG_PATH` là tùy chọn đọc file JSON **trong container/process**, mode env nếu có sẽ override mode trong file. File candidate hiện là open_close chỉ để minh họa/reproduce, không phải cấu hình đã được chọn. Offline có thể dùng `ClothingDetector(mask_config=MaskConfig(**config_dict))`.

## Lệnh tái lập từ PowerShell root

```powershell
$m2Root = (Get-Location).Path
$m2Bind = "type=bind,source=$m2Root,target=/workspace"

# Giữ tag phase1 làm đối chứng; chỉ build service hiện tại và image test riêng.
docker compose build object-detection
docker build -f object_detection/Dockerfile.tests -t fashion-m2:phase3-tests ./object_detection
docker run --rm --network none --mount $m2Bind -w /workspace -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e RUN_REAL_HSV_INTEGRATION=0 --entrypoint python fashion-m2:phase3-tests -m evaluation.run_m2_checks --output output/m2/runs/phase3_checks_repeat
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase3-tests -m evaluation.check_m2_m3_contract
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase3-tests -m evaluation.check_m2_phase3 examples --output output/m2/runs/phase3_examples_repeat
docker run --rm --network none --cpus 4 --memory 4g --mount $m2Bind -w /workspace -e YOLO_CONFIG_DIR=/tmp --entrypoint python fashion-m2:phase3-tests -m evaluation.check_m2_phase3 regression --output output/m2/runs/phase3_regression_repeat
```

`examples` sinh lại fixture JSON ở tests; dùng output mới để không ghi đè bằng chứng đã lưu. Không chạy generator trên một artifact đang được người khác chỉnh tay. Regression của lượt này dùng image CPU phase 1 với source hiện tại bind vào `/workspace`; dependencies API/full tests nằm ở image test riêng. Lock package đã lưu; không tải encoder weights.

Smoke HTTP raw đã chạy trong container tạm `fashion_m2_phase3_smoke` (network none, model read-only), sau đó xóa. Service thường `fashion_object_detection` giữ off. Khi muốn tái lập smoke enabled, tạo container tương tự với image detection hiện tại, model mount read-only và `M2_MASK_MODE=raw`, rồi chạy `evaluation.check_m2_phase3 http` qua network `container:<tên container>`; runner đọc regression.json trong cùng output directory. Không cần bật DB/main API/frontend.

## Bước tiếp theo

Phase 4: gán/kiểm nhãn bbox và mask thật, metric target/IoU và phân nhóm lỗi. Phase 5 mới chọn polarity/gate/mode bằng validation, khóa cấu hình rồi chạy test; chỉ bật default nếu bằng chứng đủ. Tiếp tục dùng HSV147 cố định, cache/candidate/ranking cố định trong ablation mask.
