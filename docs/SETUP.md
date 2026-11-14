# Chạy Fashion Visual Search

Chạy các lệnh từ root. Cần Docker Desktop, model YOLO, PostgreSQL có catalog/embedding và cache HSV để chạy đầy đủ tìm kiếm. Kiểm thử offline xem [EVALUATION](EVALUATION.md).

## Cấu hình và model

Nếu chưa có `.env`, sao chép `.env.example` rồi sửa giá trị phù hợp; không ghi đè `.env` sẵn có.

```powershell
Copy-Item .env.example .env
```

Đặt model tại `object_detection/best.pt` (dấu gạch dưới). Compose mount read-only vào `/models/object-detection/best.pt`. SHA256 model trong evidence M2: `d0d45d5299395d3a3ed0202596d3b22fa11494190a271c5b5adf429d6a9391fe`.

Cấu hình khóa: `PREPROCESS_MODE=identity`, `M2_MASK_MODE=off`. Identity vẫn xử lý EXIF/RGB/alpha/resize cạnh dài tối đa 1280; gamma và các mode mask khác dùng cho thí nghiệm.

## Detection độc lập

```powershell
docker compose up -d --build object-detection
Invoke-RestMethod http://localhost:8001/health
```

Kiểm tra `model_available`; `model_loaded` có thể false trước inference đầu. Upload bằng `POST http://localhost:8001/detect/upload`, multipart field `file`; xem [hợp đồng M2](M2_CONTRACT.md).

## Catalog và HSV

Query/catalog cần cùng embedding model/version: mặc định `patrickjohncyh/fashion-clip`, version 2, vector 512 chiều. Product ID trong PostgreSQL phải khớp cache HSV.

Với catalog đã có embedding, dùng [color-only rebuild](M3_HSV_PRODUCTION.md#rebuild-catalog-colors-without-re-embedding) để tạo cache mới; không chạy lại vectorizer chỉ để cập nhật màu. Đặt `HSV_COLOR_CACHE_FILE` trong `.env` trỏ tới file host có thật. Compose mount thành `/app/color-cache/catalog.json`, dùng profile `/app/config/hsv147.json`.

Cache chẩn đoán `docs/m3_results/catalog_color_cache.json` dùng `whole_catalog_photo_v1`, không phải production `source_roi_v1`. Không đổi nhãn cache để dùng thay. Khi bàn giao M2, cache mặc định `data/color_cache_hsv147.json` chưa có; full retrieval chưa được nghiệm thu.

## Khởi động hệ thống

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

| Thành phần | Địa chỉ host |
|---|---|
| Frontend | http://localhost:5173 |
| API / OpenAPI | http://localhost:8000/docs |
| Detection / OpenAPI | http://localhost:8001/docs |
| PostgreSQL | localhost:5433 (Compose: postgres-pgvector:5432) |

API health không thay việc kiểm DB có candidate đúng model/version và cache đủ ID. Nếu cần nạp catalog mới, đọc `db/database.py` và kiểm input trước khi chạy `python -m db.database`; entry point này khởi tạo DB và nạp dữ liệu. Migration bổ sung chạy bằng `python -m db.run_migrations` trong môi trường có cấu hình DB tương ứng.

## Lỗi thường gặp

- Detection 503: model thiếu/mount sai. `object-detection` là service; `object_detection/` là thư mục source.
- API lỗi bind mount: tạo cache HSV thật, sửa `HSV_COLOR_CACHE_FILE`, recreate container. Compose không tự tạo file cache.
- Thiếu màu: kiểm policy, coverage, config/version/ROI policy. Với `zero`, màu thiếu đóng góp 0; không có RGB fallback.
- Không có candidate: kiểm dữ liệu DB, model/version và ngưỡng lọc trước khi đổi trọng số.

Xem [bàn giao M4](M4_HANDOFF.md) để nghiệm thu đường chạy đầy đủ.
