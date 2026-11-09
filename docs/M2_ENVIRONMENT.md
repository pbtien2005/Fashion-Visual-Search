# Môi trường và cách chạy phase 1

Đã thực hiện ngày 07/11/2026, commit nền `3247a5b33a016fd064ac9851410e7ec755d5e320` cộng các thay đổi M2 chưa commit. Model tại `object_detection/best.pt`, 52.020.562 byte, SHA-256 `d0d45d5299395d3a3ed0202596d3b22fa11494190a271c5b5adf429d6a9391fe`.

## Môi trường đã chạy thật

- Docker Desktop engine 29.2.0; Linux/WSL2; image `fashion-m2:phase1` và service Compose `fashion_object_detection`.
- Python 3.12.12; Ultralytics 8.4.31, cùng version lưu trong checkpoint; torch 2.11.0+cpu; torchvision 0.26.0+cpu; NumPy 2.2.6; Pillow 11.3.0; OpenCV 4.13.0.92; FastAPI 0.115.4; Pydantic 2.9.2.
- Môi trường này dùng CPU, chưa cấu hình CUDA/GPU passthrough. Máy host đã xác nhận RTX 5050 Laptop khoảng 8 GB nhưng phase 1 không dùng GPU và không train.
- Các version trực tiếp được pin tại `object_detection/requirements-m2.txt`; bản đầy đủ đã cài tại `docs/m2/phase1/requirements.lock.txt`. Các gói torch có hậu tố `+cpu`, cần index chính thức CPU khi dựng lại từ lock.
- Model được mount read-only vào `/models/object-detection/best.pt`. `.dockerignore` loại weights khỏi image; không tạo bản sao model vào image.
- Không khởi động DB, main API, FashionCLIP hoặc frontend. Không thay đổi container ứng dụng khác trên máy.

Image đã dùng: `sha256:2041a2f15f8df30272144b3dd9ee6861382f4b2e67156c03c915521f60ae46b3` (theo `docker image inspect fashion-m2:phase1`). Snapshot hash các file M2 tại `docs/m2/phase1/source_snapshot.json`. Baseline đã lưu là code cũ của pipeline crop cộng sửa cách tìm weights, chưa có phân vùng.

**Sau phase 2:** giữ nguyên tag `fashion-m2:phase1` làm đối chứng code cũ, không build đè tag này bằng source mới. Các lệnh bên dưới ghi lại workflow ở phase 1; với code hiện tại, xem `docs/m2/phase2/RESULTS.md`. Bind repo vào image cũ sẽ chạy source mới tại `/workspace`, nên không gọi kết quả đó là baseline phase 1. Bằng chứng phase 1 đã lưu không bị ghi đè.

## Chạy lại từ PowerShell tại root repo

```powershell
docker build -t fashion-m2:phase1 ./object_detection
$m2Root = (Get-Location).Path
$m2Bind = "type=bind,source=$m2Root,target=/workspace"

# Sinh fixture; không tải model, không cần DB hoặc mạng.
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m evaluation.prepare_m2_phase1

# 11 test model path và API bootstrap, rồi 6 test fixture.
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m coverage run --source=object_detection -m unittest discover -s tests -p 'test_detection*.py' -v
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m coverage run --append --source=object_detection -m unittest discover -s tests -p test_m2_contract_fixtures.py -v

# Hai nhóm test cũ không cần môi trường API/encoder đầy đủ.
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m unittest discover -s tests -p test_color_utils.py -v
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m unittest discover -s tests -p test_benchmark_suite.py -v

# Baseline mới: thư mục đầu ra phải chưa tồn tại để giữ kết quả cũ.
docker run --rm --network none --cpus 4 --memory 4g --mount $m2Bind -w /workspace -e YOLO_CONFIG_DIR=/tmp -e YOLO_AUTOINSTALL=false --entrypoint python fashion-m2:phase1 -m evaluation.run_m2_phase1_baseline --model object_detection/best.pt --output output/m2/runs/baseline_repeat_01

# Chỉ bật detection service; model mount theo docker-compose.yml.
docker compose up -d --no-deps --build object-detection
Invoke-RestMethod http://localhost:8001/health
docker run --rm --network container:fashion_object_detection --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m evaluation.check_m2_phase1_http
```

Trước baseline cuối đã chạy một lượt có coverage để kiểm tra đường xử lý YOLO thật; lượt đó chỉ phục vụ coverage. Lượt lưu cuối `baseline_20261107_cpu` chạy **không coverage**, một blank warm-up, CPU limit 4, RAM limit 4 GiB; số thread torch thực tế 8. Blank warm-up không bảo đảm mọi shape ảnh đã warm hoàn toàn. Không dùng các timing này như GPU benchmark hoặc end-to-end latency.

Coverage 88% ghi trong progress là kết hợp test + đường chạy model thật. Chạy riêng hai lệnh unit test ở trên không tái tạo toàn bộ tỷ lệ đó; nếu cần đo cùng scope, chạy thêm runner baseline dưới `coverage run --append` vào thư mục kết quả riêng, rồi `coverage report -m`. Không dùng lượt instrumented để báo latency cuối.

## Kết quả và ranh giới kiểm chứng

- 17 test mới M2 và 7 test cũ (màu/benchmark) đã qua. Coverage module detection 88%, riêng helper mới `config.py` 100%.
- Chưa chạy hai nhóm test search/vectorizer cần dependency API/encoder ngoài image detection; không khẳng định toàn bộ test suite đã chạy.
- `http://localhost:8001/health`: `model_available=true`, `model_loaded=true` sau upload thật.
- Real HTTP upload ảnh regression trả HTTP 200 và 2 crop; JPEG decode 224×224; MIME không hợp lệ trả 400. Chi tiết tại `docs/m2/phase1/http_smoke.json`.
- Baseline 16 ảnh regression, không nhãn bbox/mask người kiểm tra. Kết quả tại `docs/m2/phase1/baseline_20261107_cpu/`; predictions không phải ground truth, không suy ra mAP/recall hoặc hiệu quả mask.
- Hợp đồng tại `docs/M2_CONTRACT.md`; ví dụ candidate response chỉ là fixture cho phase 2–3. Production hiện vẫn trả schema legacy.

## Dữ liệu cần chuẩn bị ở phase 4

`evaluation/m2/regression_manifest.json` khóa 16 ảnh outfit cũ bằng SHA-256 và giữ nguồn/tác giả/license từ metadata có sẵn. Các ảnh này đã được dùng phát triển; không được tự đổi thành test mới.

`evaluation/m2/annotation_manifest.draft.json` là bản nháp **rỗng**, có record template. Phần Người 2 cần 20 ảnh mới sau khử trùng: 5 validation, 15 test; thêm 2 mask validation và 3 mask test. Checklist:

1. Chọn một món mục tiêu/ảnh, ghi nguồn và quyền sử dụng, nhóm người/cảnh để tránh rò rỉ giữa split.
2. Gán bbox/class/màu/nhóm ánh sáng độc lập với dự đoán; xác nhận bằng người xem.
3. Khóa EXIF/transform theo M1; mask lưu full working canvas hoặc ROI có bbox/transform đầy đủ.
4. Kiểm chéo nhãn; không dùng metadata màu hoặc prediction của model làm nhãn thật.
5. Chỉ dùng validation chọn config; test giữ kín tới phase 5. Chưa có ảnh mới hoặc nhãn thì để pending.

## Dọn lượt bị gián đoạn

Đã xóa `.venv`, `tmp/python`, `tmp/python-bin`, `tmp/uv-cache` sau khi kiểm tra chúng nằm trong workspace. Dừng đúng hai tiến trình Black formatter đang dùng môi trường tạm để nhả file; không đổi cấu hình VS Code. Các test/code viết dở được hoàn thiện và chạy qua, nên giữ lại. Giữ `tmp/pdfs` và `tmp/pdf-deps` phục vụ đọc tài liệu trước đó; `tmp/` được ignore. Không xóa tài liệu gốc hoặc `best.pt`.

Tham khảo quy ước input NumPy BGR của YOLO: [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/). Cài CPU torch theo [nguồn PyTorch chính thức](https://pytorch.org/get-started/locally/). Những kết quả trong tài liệu này lấy từ lượt chạy thực tế, không từ ví dụ trong tài liệu thư viện.
