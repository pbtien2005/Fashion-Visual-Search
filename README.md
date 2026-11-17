# Fashion Visual Search

Tìm sản phẩm thời trang tương tự từ **ảnh chụp ngoài đời** (người mặc đồ, nền phức tạp). Đồ án môn Xử lý ảnh, PTIT, nhóm 13 (đề "Lựa chọn 2 – đề tài thực tế").

Ảnh truy vấn đi qua bốn bước, mỗi bước dùng kỹ thuật của môn học hoặc mở rộng từ đó:

| Bước | Việc làm | Tham số chính |
|---|---|---|
| M1. Chuẩn hóa ảnh | Sửa hướng EXIF, đổi về RGB, thu nhỏ cạnh dài tối đa 1280 px (Lanczos). Gamma 0,7 chỉ chạy khi `mean luma < 0,30` và mặc định tắt (`PREPROCESS_MODE=identity`) | `PREPROCESS_*` |
| M2. Phát hiện và cắt vùng | YOLOv8m 10 lớp quần áo, `conf=0,5`; cắt 3 biến thể (sát, +15 %, +30 %) rồi đưa về 224×224. Có sẵn phân ngưỡng Otsu có điều kiện + morphology 3×3 để tách nền, mặc định tắt (`M2_MASK_MODE=off`) | `M2_MASK_MODE` |
| M3. Đặc trưng | FashionCLIP 512 chiều (kiểu dáng) + histogram HSV 147 chiều (màu: 18 H × 8 S + 3 bin vô sắc, khoảng cách Hellinger) | `EMBEDDING_*`, `HSV_*` |
| M4. Tìm và xếp hạng | pgvector (HNSW, cosine) lấy top-100 mỗi biến thể, xếp hạng lại tối đa 20 sản phẩm: `0,60·ảnh + 0,15·loại + 0,15·màu + 0,10·text/style − phạt`, rồi áp ngưỡng chấp nhận | `RERANK_*`, `MATCH_SCORE_THRESHOLD` |

Dữ liệu sản phẩm được tạo ngoại tuyến bằng ETL: Apify cào Amazon → loại trùng theo ASIN → tải ảnh lên Cloudflare R2 → trích đặc trưng → nạp PostgreSQL.

## Cấu trúc thư mục

```text
api/               API tìm kiếm (FastAPI, cổng 8000): gọi YOLO, truy vấn pgvector, xếp hạng lại
object_detection/  Dịch vụ phát hiện (FastAPI, cổng 8001): M1 + M2 (YOLO, ROI, Otsu/morphology)
AI/                M3: FashionCLIP, histogram HSV147, vectorizer, dựng cache màu
db/                Schema PostgreSQL + pgvector, nạp dữ liệu, migration
etl/               Thu thập dữ liệu: Apify (Amazon) → R2
web/fe/            Giao diện React + Vite (cổng 5173)
evaluation/        Bộ đánh giá: benchmark 75 ảnh, nhãn, công cụ gán nhãn và chấm
samples/           8 ảnh mẫu ngoài đời (Wikimedia Commons) để thử hệ thống, kèm nguồn và giấy phép
tests/             Kiểm thử đơn vị và tích hợp (fixture, không cần mạng)
data/              Catalog mẫu 28 sản phẩm + cache màu (bộ đủ 1.752 sản phẩm không có trong repo)
docs/              Tài liệu kỹ thuật; kết quả benchmark ở docs/benchmark/
```

## Yêu cầu môi trường

| Thành phần | Phiên bản đã dùng |
|---|---|
| Hệ điều hành | Windows 11 (Docker Desktop); Linux/macOS dùng được tương tự |
| Docker + Docker Compose | Docker 28.4 (cách chạy khuyến nghị) |
| Python (chỉ khi chạy test/benchmark ngoài Docker) | 3.12 (image) hoặc 3.13 (máy phát triển) |
| Node.js (chỉ khi chạy giao diện ngoài Docker) | 22 |
| Phần cứng | CPU, không cần GPU. RAM từ 8 GB trở lên; ảnh đầu tiên chậm hơn vì nạp model |
| Mạng | Lần đầu cần tải FashionCLIP từ Hugging Face (`patrickjohncyh/fashion-clip`) và torch CPU |

Thư viện chính: `torch 2.11.0+cpu`, `transformers 4.57`, `ultralytics 8.4`, `fastapi 0.115`, `psycopg 3` + `pgvector`, `opencv`, `pillow`, `numpy`. Danh sách đầy đủ: [requirements.txt](requirements.txt) (API), [object_detection/requirements-m2.txt](object_detection/requirements-m2.txt) (phát hiện), [requirements-benchmark.txt](requirements-benchmark.txt) (benchmark và test).

## Dữ liệu và model cần tải

Repo không chứa model và catalog đầy đủ do dung lượng. Chuẩn bị theo bảng:

| Thứ cần có | Đặt tại | Nguồn |
|---|---|---|
| Checkpoint YOLOv8m (`best.pt`, ~52 MB, SHA-256 `d0d45d52…9391fe`) | `object_detection/best.pt` (Docker) và `models/object-detection/best.pt` (benchmark) | [Google Drive của nhóm](https://drive.google.com/drive/folders/1KMOuhe7r0Vs_m1i7EKYlOi1IRJ7zaZLC?usp=sharing) |
| Catalog đầy đủ 1.752 sản phẩm Amazon (có embedding) | `data/fashion_with_vectors.json` (benchmark) | [Google Drive của nhóm](https://drive.google.com/drive/folders/1KMOuhe7r0Vs_m1i7EKYlOi1IRJ7zaZLC?usp=sharing) (tệp `fashion_with_vectors.json`); hoặc tự tạo bằng mục "Tạo lại catalog" |
| Catalog mẫu 28 sản phẩm + cache màu HSV147 | `data/fashion_with_hsv_vectors.json`, `data/color_cache_hsv147.json` | Đã có sẵn trong repo, đủ để chạy demo |
| Ảnh truy vấn benchmark (75 ảnh có nhãn + 88 ảnh ứng viên) | `evaluation/queries/bench/` | Có sẵn trong repo. Nguồn, tác giả, giấy phép từng ảnh: [candidates.csv](evaluation/queries/bench/candidates.csv) |
| Ảnh Wikimedia Commons bổ sung | `evaluation/queries/real_outfits/` | Có sẵn trong repo. Nguồn: [real_outfit_sources.md](evaluation/real_outfit_sources.md) |
| Ảnh sản phẩm Amazon | Cloudflare R2 (`images/{ASIN}.jpg`) | Tải bằng ETL; xem lưu ý giấy phép ở mục Giới hạn |

Ảnh truy vấn lấy từ Wikimedia Commons, giữ nguyên tác giả và giấy phép (CC0, CC BY, CC BY-SA hoặc public domain). Số liệu: validation 23 ảnh, test 52 ảnh, cộng 266 ảnh cho phép thử độ bền.

## Tạo lại catalog (tùy chọn)

Chỉ cần khi muốn tự dựng catalog thay vì tải sẵn. Cần `APIFY_API_TOKEN` và các biến `R2_*` trong `.env`; Apify tính phí theo lượng cào, và kết quả thay đổi theo thời điểm vì dữ liệu Amazon đổi liên tục, nên không đảm bảo trùng 1.752 sản phẩm đã dùng.

```powershell
python -m etl.pipeline        # cào Amazon, loại trùng theo ASIN, tải ảnh lên R2, ghi data/fashion_metadata.json
python -m AI.vectorizer       # cắt vùng sản phẩm, FashionCLIP 512 chiều + HSV147, ghi data/fashion_with_hsv_vectors.json
python -m db.database         # nạp PostgreSQL
```

Từ khóa và số sản phẩm mỗi lượt nằm ở dòng cuối `etl/pipeline.py` (mặc định `"men graphic t-shirt"`, 15 sản phẩm); đổi rồi chạy lặp lại để tích lũy. `AI.vectorizer` từ chối ghi đè kết quả cũ, nên đổi tên hoặc chuyển bộ mẫu 28 sản phẩm trong `data/` đi chỗ khác trước khi chạy.

## Cài đặt và chạy demo (Docker)

Chạy tại thư mục gốc. Dùng PowerShell trên Windows.

**1. Cấu hình**

```powershell
Copy-Item .env.example .env
```

Giữ nguyên giá trị mặc định để chạy demo. Các khóa cấu hình đã chốt: `PREPROCESS_MODE=identity`, `M2_MASK_MODE=off`. Biến `R2_*` và `APIFY_API_TOKEN` chỉ cần khi chạy ETL hoặc hiển thị ảnh sản phẩm từ R2.

**2. Đặt model** tại `object_detection/best.pt` (xem bảng trên).

**3. Khởi động**

`docker-compose.override.yml` là file chỉnh cổng cục bộ của người phát triển, không có trong repo. Nếu máy bạn có file này, thêm `-f docker-compose.yml` vào mọi lệnh `docker compose` bên dưới để dùng cấu hình chuẩn.

```powershell
docker compose -f docker-compose.yml up -d --build
docker compose -f docker-compose.yml ps
```

Lần build đầu tải torch CPU nên mất vài phút. Khi cả 4 dịch vụ ở trạng thái `healthy`:

| Dịch vụ | Địa chỉ |
|---|---|
| Giao diện | http://localhost:5173 |
| API (OpenAPI) | http://localhost:8000/docs |
| Dịch vụ phát hiện (OpenAPI) | http://localhost:8001/docs |
| PostgreSQL | localhost:5433 |

**4. Nạp catalog mẫu vào PostgreSQL** (tạo bảng, bật `pgvector`, nạp 28 sản phẩm từ `data/fashion_with_hsv_vectors.json`). Chạy một lần trên máy chủ, cần Python với `requirements.txt`:

```powershell
$env:DB_HOST = "localhost"; $env:DB_PORT = "5433"
python -m db.database
```

**5. Kiểm tra sức khỏe**

```powershell
Invoke-RestMethod http://localhost:8000/health
```

**6. Tìm kiếm bằng ảnh.** Mở http://localhost:5173, tải một ảnh trong [samples/](samples/README.md) lên (JPEG/PNG, tối đa 8 MB). Hoặc gọi API:

```powershell
curl.exe -s -F "file=@samples/sample_01_vest_dress.jpg" "http://localhost:8000/api/search/upload?top_k=5&mode=hybrid"
```

`mode` nhận `hybrid` (mặc định), `embedding_only` hoặc `histogram_only`; `top_k` từ 1 đến 20.

Dừng hệ thống: `docker compose -f docker-compose.yml down` (thêm `-v` để xóa cả dữ liệu PostgreSQL).

## Đầu ra mong đợi

`/health` trả về `{"success": true, "status": "ok", ...}` và `object_detection.available = true`.

`/api/search/upload` trả JSON, mỗi món quần áo được phát hiện là một phần tử trong `detected_items`:

```json
{
  "success": true,
  "detected_items": [
    {
      "class_name": "short_sleeve_dress",
      "class_name_vi": "…",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2],
      "crop_image_base64": "…",
      "query_color_name": "black",
      "is_confident_match": true,
      "similar_products": [
        {
          "id": "B0…",
          "title": "…",
          "category": "dress",
          "image_similarity": 0.81,
          "category_score": 1.0,
          "color_score": 0.9,
          "text_style_score": 0.4,
          "final_score": 0.78,
          "product_url": "https://www.amazon.com/…",
          "best_crop_variant": "tight"
        }
      ],
      "message": "…"
    }
  ]
}
```

Các số trong ví dụ chỉ minh họa cấu trúc. Điều cần kiểm tra:

- Ảnh có người mặc đồ thì `detected_items` không rỗng, mỗi phần tử có `bbox`, lớp và danh sách sản phẩm sắp theo `final_score` giảm dần.
- Ảnh không có quần áo trả `detected_items: []` kèm `message`.
- `is_confident_match = false` nghĩa là điểm thấp hơn ngưỡng chấp nhận; sản phẩm gần nhất nằm ở `low_confidence_products`.
- Giao diện hiển thị ảnh gốc, vùng cắt và các sản phẩm gợi ý kèm điểm.

Catalog mẫu chỉ có 28 sản phẩm thuộc 7 nhóm (áo, áo khoác, đầm, quần, quần short, váy, áo dài tay), nên kết quả demo chủ yếu để xem luồng chạy, không phản ánh chất lượng thật. Chất lượng đo trên catalog đủ 1.752 sản phẩm, xem mục Kết quả.

## Chạy kiểm thử

Môi trường Python riêng, cài thư viện benchmark (giữ torch/torchvision CPU như ghi trong file):

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Kết quả mong đợi: toàn bộ test qua (lần đo gần nhất: 171 test, 170 qua, 1 bỏ qua do cần model thật; độ phủ 93 % cho module M2). Test dùng fixture, không cần mạng, model hay PostgreSQL. Chi tiết: [docs/EVALUATION.md](docs/EVALUATION.md).

## Chạy lại benchmark

Cần `models/object-detection/best.pt` và `data/fashion_with_vectors.json` (mục Dữ liệu). Lần đầu FashionCLIP được tải về Hugging Face cache. Chạy chế độ nhanh 3 ảnh để kiểm tra môi trường:

```powershell
$run = "evaluation/results/bench_reproduce"
.\venv\Scripts\python.exe -m evaluation.benchmark.manifest check
.\venv\Scripts\python.exe -m evaluation.benchmark.run validation --run-dir $run --smoke 3
```

Chạy đầy đủ (validation, rồi test một lần sau khi khóa cấu hình, và độ bền):

```powershell
.\venv\Scripts\python.exe -m evaluation.benchmark.run validation --run-dir $run
.\venv\Scripts\python.exe -m evaluation.benchmark.run test --run-dir $run
.\venv\Scripts\python.exe -m evaluation.benchmark.build_perturbed
.\venv\Scripts\python.exe -m evaluation.benchmark.run robustness --run-dir $run
```

Đầu ra nằm trong `$run/{validation,test}/`, gồm `summary.md` (bảng số liệu), `per_query.csv` và kết quả từng truy vấn; tập test chỉ chạy một lần sau khi khóa. Bản kết quả đã công bố ở [docs/benchmark/main](docs/benchmark/main), mô tả đầy đủ ở [evaluation/BENCHMARK.md](evaluation/BENCHMARK.md).

## Kết quả

Đo trên 75 ảnh ngoài đời (23 validation, 52 test, catalog 1.752 sản phẩm, CPU). HitRate@5 là tỷ lệ ảnh có ít nhất một sản phẩm chấp nhận được trong 5 hạng đầu; mẫu số là các ảnh có sản phẩm khớp (test: 47 ảnh).

| Cấu hình (tập test) | HitRate@5 | MRR@5 |
|---|---:|---:|
| A0, cả ảnh, không phát hiện | 0,340 | 0,214 |
| C1p, đối chứng cổ điển (màu trội + từ khóa) | 0,213 | 0,128 |
| C2, chỉ embedding trên vùng YOLO | 0,362 | 0,266 |
| **B0, pipeline đầy đủ (hybrid)** | **0,426** | **0,363** |

Phát hiện đúng lớp với IoU ≥ 0,5 đạt 0,519 (27/52) ở test và 0,739 (17/23) ở validation; đây là nút thắt chính. Chênh lệch giữa B0 và C2 chỉ 3 ảnh trên 47, nên chỉ coi là xu hướng.

## Giới hạn cần biết

- Nhãn độ phù hợp của sản phẩm do AI chấm; người rà lại 25/75 ảnh (đồng ý 90,9 %, kappa 0,82). Tập test không hoàn toàn độc lập vì mẫu nhỏ và nguồn ảnh chung.
- Mask Otsu chưa đạt ngưỡng IoU 0,8 nên mặc định tắt; B2, B3, B4, C1 (qua API) và cấu hình cuối F chưa đo.
- Ngưỡng chấp nhận 0,70 trong `.env.example` loại gần hết kết quả. Benchmark khuyến nghị T = 0,45 và ảnh tối thiểu 0,40; giá trị này chưa được áp làm mặc định.
- Catalog Amazon chưa làm sạch và lệch theo từ khóa cào. Ảnh sản phẩm thuộc Amazon: chỉ dùng cho mục đích học tập, không phân phối lại.

## Tài liệu và nhóm thực hiện

Tài liệu kỹ thuật: [docs/README.md](docs/README.md), [cài đặt chi tiết](docs/SETUP.md), [bàn giao tích hợp](docs/M4_HANDOFF.md), [kiểm thử](docs/EVALUATION.md), [nguồn dữ liệu](docs/DATA_SOURCES.md).

Nhóm 13: Phạm Bá Tiến (tiền xử lý, thu thập dữ liệu, cơ sở dữ liệu vector, benchmark), Triệu Tuấn Anh (YOLO, ROI, Otsu, morphology), Nguyễn Minh Vinh (FashionCLIP, HSV147), Nguyễn Thành Đạt (API, giao diện, xếp hạng lại). Khai báo sử dụng AI nằm trong báo cáo nộp kèm.
