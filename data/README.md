# Catalog mẫu cho M4

Bộ dữ liệu này có 28 ảnh từ `evaluation/queries/starter_catalog/`. Các tên sản phẩm là nhãn demo suy ra từ loại trang phục và ID. Cache màu dùng ảnh gốc và ghi nhận thiếu mask cho cả 28 ảnh.

| File | Nội dung |
|---|---|
| `fashion_metadata.json` | ID, nhãn demo, nhóm trang phục và đường dẫn ảnh |
| `fashion_with_hsv_vectors.json` | Cùng 28 ID, kèm embedding FashionCLIP 512 chiều và descriptor HSV147 |
| `color_cache_hsv147.json` | Cache màu cho API/M4, cùng dữ liệu với `output/hsv.json` |

Từ thư mục gốc repo, với dependencies Python đã cài, chạy trên cơ sở dữ liệu demo mới. Loader cập nhật các dòng trùng ID nếu bảng đã có dữ liệu.

```powershell
docker compose up -d --wait postgres-pgvector
$env:DB_HOST = 'localhost'
$env:DB_PORT = '5433'
$env:DB_USER = 'pbtien'
$env:DB_PASSWORD = 'change_me'
$env:DB_NAME = 'fashion_db'
python -m db.database
docker compose exec -T postgres-pgvector psql -U pbtien -d fashion_db -c 'SELECT count(*), min(vector_dims(embedding)) FROM fashion_items;'
```

Trên database mới, kết quả mong đợi là 28 dòng và 512 chiều. API Docker mặc định mount `data/color_cache_hsv147.json`. Nếu chạy API trên host, dùng `HSV_COLOR_CACHE_PATH=data/color_cache_hsv147.json`. Chạy tìm kiếm qua ảnh upload còn cần detector với weights YOLO và encoder FashionCLIP. Xem [hướng dẫn production](../docs/M3_HSV_PRODUCTION.md).
