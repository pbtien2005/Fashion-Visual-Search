# 👗👔 Fashion Visual Search

Hệ thống Tìm kiếm Sản phẩm Thời trang bằng Hình ảnh (Visual Search) dựa trên mô hình AI nhúng (CLIP Model) và cơ sở dữ liệu Vector (PostgreSQL + pgvector).

## 🌟 Tính năng chính

- **Cào dữ liệu tự động (ETL)**: Thu thập thông tin sản phẩm (Tên, Giá, Hãng, Hình đại diện) từ Amazon thông qua mạng lưới Apify Actor.
- **Lưu trữ tĩnh hiệu suất cao**: Tự động tải ảnh sản phẩm gốc và đẩy trực tiếp lên Cloudflare R2 nhằm tạo kho ảnh nội bộ độc lập.
- **Trích xuất Vector AI**: Ứng dụng mạng nơ-ron đa thể thức OpenAI CLIP (`clip-vit-base-patch32`) để biến ảnh sản phẩm thành mảng 512 con số (Vector Embeddings) có ngữ nghĩa.
- **Tìm kiếm tương đồng (Similarity Search)**: Dựa vào một hình ảnh bất kỳ (Input Image), máy sẽ chuyển đổi thành vector, sau đó sử dụng **PostgreSQL với pgvector HNSW Index** để tìm và tính toán khoảng cách Cosine (Cosine Distance). Trả về các sản phẩm có phong cách, kiểu dáng giống nhất ở tốc độ phần nghìn giây.

## 🏗️ Kiến trúc Hệ thống

1. **ETL Pipeline (`etl/`)**: Apify ➔ Cloudflare R2 ➔ Lọc & Trích xuất JSON cục bộ.
2. **AI Engine (`AI/vectorizer.py`)**: Đọc kho JSON ➔ Tải ảnh vào RAM ➔ Extract bằng Torch & HuggingFace ➔ Ghi Vectors vào JSON.
3. **Database (`db/` & Docker)**: Khởi tạo bảng, index HNSW, đẩy đống Vectors vào PostgreSQL cục bộ.
4. **Searcher (`searcher.py`)**: Người dùng chạy script ➔ Query SQL pgvector ➔ Hiển thị kết quả.

## 🛠 Cài đặt & Chuẩn bị

### 1. Yêu cầu hệ thống
- Python 3.10+
- Docker Desktop (để chạy database)
- Môi trường ảo (venv)

### 2. Cài đặt thư viện Python
```powershell
# Chạy ở thư mục gốc của dự án
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường (`.env`)
Tạo một file `.env` ở thư mục gốc theo format:
```env
# APIFY Setup (Dùng crawler lấy data)
APIFY_API_TOKEN=your_apify_token_here

# CLOUDFLARE R2 Setup (Dùng lưu trữ ảnh)
R2_ENDPOINT_URL=https://<your_account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=fashion-visual-search

# POSTGRESQL Setup (Khai báo đúng như trong docker-compose.yml)
DB_HOST=localhost
DB_PORT=5433
DB_USER=pbtien
DB_PASSWORD=tien12345
DB_NAME=fashion_db
```

### 4. Khởi động PostgreSQL (pgvector)
```powershell
docker-compose up -d
```
Bạn có thể xác nhận container đang chạy ở port `5433`.

## 🚀 Hướng dẫn Chạy Pipeline Toàn Tập

### Bước 1: Thu thập Dữ liệu (ETL)
Chạy script để crawl dữ liệu từ khóa thời trang và lưu ảnh sang R2. Thông tin sẽ được xuất ra `data/fashion_metadata.json`.
```powershell
python etl/test.py
```
*(Bạn có thể cấu hình danh sách từ khóa trực tiếp bằng cách sửa file `etl/test.py`)*

### Bước 2: AI Vectorizer
Đưa ảnh qua model AI để tạo vector. Dữ liệu sẽ lưu ở `data/fashion_with_vectors.json`.
```powershell
python AI/vectorizer.py
```

### Bước 3: Nạp dữ liệu vào Database
Tạo schema trên PostgreSQL và Import đống sản phẩm đã nhúng vector vào CSDL.
```powershell
python db/database.py
```

### Bước 4: Chạy Tìm Kiếm (Visual Search)
Bạn nhập URL bất kỳ của một tấm ảnh ở dòng `test_image_url` bên trong `searcher.py`. Sau đó chạy lệnh:
```powershell
python searcher.py
```
Hệ thống sẽ in ra top những sản phẩm thời trang giống tấm hình của bạn nhất!

## 🔖 Công nghệ sử dụng
- **Python**: Ngôn ngữ xử lý mạch chính.
- **HuggingFace (`transformers`) & PyTorch**: Chạy mô hình pretrained **CLIP**.
- **Apify Client**: Agent đám mây đi cào mảng E-Commerce.
- **Boto3**: Tương tác với Cloudflare R2 theo chuẩn API S3.
- **PostgreSQL 16 & pgvector & psycopg 3**: Cấu trúc Data Warehouse kết hợp Index HNSW cho việc tìm kiếm Vector siêu tốc.
- **Docker Compose**: Đóng gói CSDL sạch sẽ không cấu hình nhọc sức.
