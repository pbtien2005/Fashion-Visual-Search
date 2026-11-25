# Benchmark model embedding

So sánh FashionCLIP với 6 model embedding khác trên bộ benchmark có sẵn: test 52 ảnh (47 `has_match`) và validation 23 ảnh. Pipeline trong `evaluation/benchmark` được giữ nguyên, chỉ thay model embedding. Kết quả và biểu đồ được ghi vào `reports/`. Theo dõi bằng notebook [benchmark_models.ipynb](benchmark_models.ipynb).

| Nhóm | Key | Model | Chiều | Cấu hình chạy |
|---|---|---|---:|---|
| Mốc | `fashionclip` | `patrickjohncyh/fashion-clip` | 512 | A0, C2, B0, C3_selected |
| Kiểm tra | `fashionclip_stored` | Code production và vector gốc của catalog | 512 | như trên |
| 1 | `marqo_fashionclip` | `Marqo/marqo-fashionCLIP` | 512 | như trên |
| 1 | `clip_b16` | `openai/clip-vit-base-patch16` | 512 | như trên |
| 1 | `laion_b32` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 512 | như trên |
| 2 | `marqo_fashionsiglip` | `Marqo/marqo-fashionSigLIP` | 768 | như trên |
| 2 | `siglip_b16` | `google/siglip-base-patch16-224` | 768 | như trên |
| 2 | `dinov2_base` | `facebook/dinov2-base` | 768 | A0, C2 (không có text encoder) |

## Phần giữ nguyên và phần thay đổi

Các phần sau được import từ `evaluation/benchmark`, không chép lại: manifest, chuẩn hóa M1, code crop của `ClothingDetector` (tight/padded/wide, 224 px, nền trắng), định nghĩa cấu hình, rerank production, ngưỡng, ghép mục tiêu và `score.config_rows`.

Mỗi model chỉ thay ba thứ:
- Vector ảnh catalog: nhúng lại toàn bộ sản phẩm từ **đúng ảnh đã tạo vector gốc**. Catalog được vector hóa ngày 30/04/2026 với `USE_PRODUCT_CROP_EMBEDDING` bật: 1709/1752 sản phẩm dùng crop YOLO "padded" (lưu ở `product_crop_bbox`), 43 sản phẩm còn lại dùng ảnh gốc. `catalog_crops.py` dựng lại crop đó (JPEG q90 → mở rộng 15 % → pad vuông nền trắng → 224 px INTER_AREA → JPEG). Kiểm tra: FashionCLIP trên crop dựng lại cho cosine ≥ 0,998 với vector lưu trong catalog ở cả 1752 sản phẩm. Nếu nhúng ảnh gốc thì median chỉ là 0,90, tức sai pipeline.
- Vector truy vấn.
- Vector text/style trong rerank B0: dùng text encoder của chính model đó.

Màu HSV của sản phẩm vẫn tra theo khóa FashionCLIP, vì màu không phụ thuộc model embedding. Repo chỉ có cache màu của 28 sản phẩm mẫu (`data/color_cache_hsv147.json`), nên `build_color_cache.py` dựng lại cache cho đủ 1.752 sản phẩm bằng `AI/rebuild_color_cache.py` nguyên bản. Nếu thiếu cache này, tên màu của sản phẩm trong câu text/style bị trống và thứ hạng B0/C3 bị lệch. `run_models.py` tự dùng cache này khi chưa đặt `HSV_COLOR_CACHE_PATH`.

Cache mẫu 28 sản phẩm được dựng từ cả ảnh, không qua YOLO; dựng lại bằng `--skip-detection` thì khớp 28/28. Tuy vậy, chính cache dùng ROI YOLO (mặc định của production) mới tái lập được benchmark đã công bố. So top-5 của `fashionclip_stored` trên tập test với `docs/benchmark/main/test/per_query.csv`:

| Cache màu | B0 trùng top-5 | C3 trùng top-5 | B0 HitRate / MRR |
|---|---|---|---|
| Không có (chỉ 28 sản phẩm) | 26/52 | 28/52 | 20 / 0,345 |
| Cả ảnh (`--skip-detection`) | 36/52 | 35/52 | — |
| **ROI YOLO (đang dùng)** | **47/52** | **47/52** | **20 / 0,360** (công bố 20 / 0,363) |

5 ảnh còn lệch cũng chính là các ảnh lệch ở C2. Ở đó hai sản phẩm có điểm gần bằng nhau (chênh khoảng 0,003–0,01) đổi chỗ, do bản công bố chạy trên Windows/CPU còn ở đây chạy Linux/GPU.

## Cách chạy

```bash
# 1. Môi trường conda riêng (không đụng base): torch CUDA + model embedding + Jupyter
bash evaluation_new/setup_env.sh

# 2. YOLO một lần trong image Docker có sẵn (không build): evaluation_new/cache/detections.json
evaluation_new/docker_run.sh detect python -m evaluation_new.detect_cache

# 3. Ảnh catalog từ R2 trong image api có sẵn: evaluation_new/cache/catalog_images/
#    Cần data/fashion_with_vectors.json (1.752 sản phẩm, Google Drive của nhóm) và khóa R2 trong .env
evaluation_new/docker_run.sh api python -m evaluation_new.fetch_catalog_images

# 3b. Cache màu HSV147 cho đủ 1.752 sản phẩm: evaluation_new/cache/color_cache_hsv147_full.json
#     Chạy AI/rebuild_color_cache.py nguyên bản; dịch vụ YOLO dùng code object_detection/ hiện tại.
docker network create fvs-bench
docker run -d --rm --name fvs-bench-detect --network fvs-bench --network-alias object-detection \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e YOLO_CONFIG_DIR=/tmp -e M2_MASK_MODE=off -e PREPROCESS_MODE=identity \
  -e OBJECT_DETECTION_MODEL_PATH=/models/object-detection/best.pt \
  -v "$PWD/object_detection:/app:ro" -v "$PWD/object_detection/best.pt:/models/object-detection/best.pt:ro" \
  fashion-visual-search-object-detection:latest
BENCH_NETWORK=fvs-bench evaluation_new/docker_run.sh api python -m evaluation_new.build_color_cache
evaluation_new/docker_run.sh api python -m evaluation_new.build_color_cache --verify
docker stop fvs-bench-detect

# 4. Chạy benchmark (có thể chạy từ notebook)
conda run --no-capture-output -n fvs-bench python -m evaluation_new.run_models --models all

# 5. Mở notebook, chọn kernel "Python (fvs-bench)", Run All -> reports/
conda run --no-capture-output -n fvs-bench jupyter lab evaluation_new/benchmark_models.ipynb
```

Mỗi model đã có `results.json` sẽ được bỏ qua. Thêm `--force` để chạy lại.

## Benchmark v1: thêm nhóm kết quả theo ảnh nguyên

v0 có 22/47 ảnh test trượt ngay từ YOLO. v1 (`benchmark_v1.py`, `benchmark_v1.ipynb`, `reports/v1/`) thêm các cấu hình dự phòng:

- `F1_B0_nodet`: dùng ảnh nguyên khi YOLO không trả hộp nào. Thực tế YOLO luôn trả ít nhất một hộp (75/75 ảnh), nên cấu hình này **không bao giờ kích hoạt**.
- `F2_B0_whole`, `F2_C2_whole`: cộng thêm một nhóm "ảnh nguyên" (cosine như A0, qua ngưỡng production). Nếu không có nhóm YOLO nào khớp món cần tìm, người dùng được giả định sẽ mở nhóm này.

Quy trình:
1. Chạy validation trước và ghi `evaluation_new/locked_v1.json` gồm định nghĩa, luật chọn và số validation.
2. Sau đó mới chạy test.

Lưu ý: ý tưởng xuất phát từ phân tích lỗi trên tập test v0, nên số test v1 chỉ mang tính xác nhận.

```bash
conda run --no-capture-output -n fvs-bench python -m evaluation_new.benchmark_v1 --stage validation
conda run --no-capture-output -n fvs-bench python -m evaluation_new.benchmark_v1 --stage test
# hoặc mở evaluation_new/benchmark_v1.ipynb, Run All -> reports/v1/
```

## Đọc kết quả

- **Thiên lệch nhãn:** nhãn độ phù hợp chủ yếu phủ các sản phẩm mà FashionCLIP từng xếp cao. Sản phẩm chưa chấm bị tính là trượt, nên chỉ số nghiêm ngặt (giống benchmark cũ) thiên vị FashionCLIP. Notebook báo thêm `unjudged_at_5` và `judged_hit_at_5` / `judged_mrr_at_5`: bỏ sản phẩm chưa chấm khỏi top-10 rồi lấy 5 sản phẩm đầu. Muốn kết luận chắc chắn thì cần chấm thêm các sản phẩm mới lọt top-5 (`evaluation/benchmark/grade.py`).
- **Cỡ mẫu nhỏ:** 47 truy vấn, chênh 1 truy vấn tương ứng 0,021 HitRate. Xem p của kiểm định dấu trong `reports/paired_vs_fashionclip.csv`.
- **Chỉ số sau ngưỡng không so sánh được:** ngưỡng 0,70/0,60 được chỉnh cho FashionCLIP, nên chỉ so chỉ số trước ngưỡng.
- **YOLO giống nhau ở mọi model**, nên recall phát hiện không đổi giữa các model.
- **`fashionclip_stored`** phải tái lập số đã công bố (test B0 20/47, C2 17/47, A0 16/47). Nếu lệch thì harness có vấn đề, chưa nên đọc các số còn lại.

## Tệp

| Tệp | Vai trò |
|---|---|
| `models.py` | Danh sách model và bộ nạp (HF CLIP, open_clip, SigLIP, DINOv2) |
| `detect_cache.py` | YOLO trong Docker, lưu box thô |
| `fetch_catalog_images.py` | Tải ảnh catalog từ R2 trong Docker |
| `benchmark_v1.py`, `benchmark_v1.ipynb`, `locked_v1.json` | Benchmark v1 (cấu hình dự phòng ảnh nguyên), khóa trên validation |
| `build_color_cache.py` | Dựng cache màu HSV147 cho đủ catalog bằng script production |
| `catalog_crops.py` | Dựng lại crop sản phẩm đã dùng để tạo vector catalog gốc |
| `run_models.py` | Nhúng catalog, phát lại box YOLO, chạy pipeline cũ cho từng model |
| `scoring.py` | Tính điểm bằng scorer cũ, thêm chỉ số chỉ tính sản phẩm đã chấm và so từng truy vấn |
| `plots.py` | Biểu đồ PNG cho `reports/` |
| `docker_run.sh` | Chạy lệnh trong image dự án có sẵn với UID của bạn |
| `setup_env.sh`, `requirements.txt` | Conda env `fvs-bench` |
