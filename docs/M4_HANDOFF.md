# Bàn giao tích hợp cho M4

Trạng thái ngày 14/11/2026: M1, M2 và M3 đã nối trong source; M2 giữ `identity + mask off`. Kiểm thử integration và detection thật có evidence. Retrieval xuyên suốt với PostgreSQL, encoder và catalog HSV production **chưa được đo**; M4 cần hoàn tất trước khi công bố chất lượng tìm kiếm.

Đọc [SETUP](SETUP.md) → [M2_CONTRACT](M2_CONTRACT.md) → [M3_HSV_PRODUCTION](M3_HSV_PRODUCTION.md) → [EVALUATION](EVALUATION.md).

## Luồng hiện tại

Upload → M1 EXIF/RGB/resize một lần → YOLO trên nhánh detection → M2 source ROI PNG và JPEG encoder variants → M3 màu từ source ROI, embedding từ JPEG → M4 cosine DB và rerank.

| Ranh giới | Code | Yêu cầu tích hợp |
|---|---|---|
| M1 → M2 | `object_detection/image_preprocessing.py`, `detector.py` | Source trước gamma; geometry working chung cho bbox/ROI/mask |
| M2 → M3 | `object_detection/schemas.py`, `utils.py`, `AI/features.py` | Canonical PNG ưu tiên, mask null hợp lệ, giữ garment ID |
| M3 → M4 | `AI/production_color.py`, `color_cache.py`, `api/search_service.py` | Query/catalog khớp config/model/version/ROI policy |
| API | `api/main.py`, `api/search_service.py` | Giữ schema/frontend và API legacy |

`source_crop_png_base64` dùng cho màu; `image_base64`/`crop_variants` JPEG dùng cho encoder. Không thay ROI màu bằng ảnh gamma/padded/render giao diện. `garment_id` nối kết quả trong request, không phải ID tracking bền vững.

M3 ưu tiên canonical source PNG/mask; vẫn hỗ trợ alias `source_roi_base64`/`mask_base64`. Canonical mask null thắng alias; source canonical hỏng không âm thầm đổi JPEG. Metadata M2 giữ dưới `color.metadata.m2_roi_metadata`.

## Cấu hình chốt

| Mục | Giá trị |
|---|---|
| M1 | `PREPROCESS_MODE=identity`, max side 1280 |
| M2 | `M2_MASK_MODE=off` |
| Embedding | `patrickjohncyh/fashion-clip`, version 2, 512 chiều L2 |
| HSV | 18×8 + Black/Gray/White = 147 chiều |
| Color version | `hsv-hs-v1-147-b0681e3cbf842188` |
| Production ROI policy | `source_roi_v1` |
| Rerank mặc định | image/category/color/text-style = .60/.15/.15/.10 |

Otsu có điều kiện và morphology nhẹ đã triển khai nhưng không đạt tiêu chí mask trên mọi reference validation, nên giữ off. Mask chỉ tác động nhánh màu, không thay encoder crop.

Bộ review dùng chung có target recall 16/20 (validation 5/5, test 11/15), 5 reference mask do AI gợi ý/người dùng kiểm. Color similarity validation n=2: off .7022, raw .8217, open_close .8257; test n=3: .4922/.5483/.5471. Mask IoU raw/open_close validation .7163/.7221, test .3035/.3069. Đây là số đo descriptor/mask theo reference, không phải Recall@K sản phẩm. Test đã xem ở phase 4, không phải holdout độc lập.

Evidence: [protocol](m2/phase5/protocol.json), [locked config](m2/phase5/locked_config.json), [final config](m2/phase5/mask_config.final.json), [review2](m2/phase4/review2/results.json), [cache audit](m2/phase5/cache_audit.json). HTTP evidence nằm trong `m2/phase5/runtime_smoke/`.

## Việc M4 cần hoàn tất

1. Tạo cache production đầy đủ cho cùng product ID trong DB. Cache mẫu 28 ảnh ở `m3_results/` dùng whole-photo policy; không thay được production. Xây file mới, kiểm coverage/failures/ROI fallback rồi cấu hình mount.
2. Kiểm query/catalog cùng model/version/config HSV/ROI policy; candidate retrieval giữ lọc model/version. Cache sai identity phải bị từ chối, không relabel mẫu.
3. Upload thật qua API với encoder, YOLO, PostgreSQL thật; kiểm garment ID, source ROI, top-K và color metadata. Thử không detection, màu thiếu, cache sai version.
4. Gán acceptable product IDs trên tập truy vấn phù hợp, tách validation/test trước tune; đo Recall@5/MRR/latency và coverage. Không suy retrieval từ mask IoU hay màu gần nhau.
5. Lưu cấu hình/evidence của lần chạy; không lấy benchmark lịch sử trước HSV làm số đo hiện tại.

`HSV_MISSING_POLICY=zero`: màu thiếu đóng góp 0, không phân phối lại trọng số màu; normalize khi thiếu text/style hiện hữu vẫn giữ. `error` từ chối màu thiếu/sai. Không có RGB fallback, `color_penalty` legacy bằng 0. Hellinger similarity đã trong [0,1]; không tự rescale/tune từ 5 mask reference.

RGB/name trả frontend là swatch/nhãn xấp xỉ suy từ HSV. H-S không giữ V của màu chromatic nên navy/brown/tan không tái tạo trung thực; không dùng swatch làm feature hoặc căn cứ chấm retrieval.

## Phạm vi bằng chứng

Suite offline M2 phase 6 đã chạy 171 test: 170 pass, 1 pretrained opt-in skip. Fixture/ASGI kiểm code/serialization, không chứng minh live DB retrieval. HTTP detection thật đã kiểm PNG round-trip và ảnh hỏng trả 400. Evidence pretrained M3 riêng là lịch sử có transport M2/DB giả lập, không phải full end-to-end thật.

20 target dùng chung từ M1 (15 ảnh M2 dev + 5 ảnh M1 test), không phải 20 ảnh mới Người 2 đóng góp. Xem [DATA_SOURCES](DATA_SOURCES.md). Đánh giá chéo độc lập và yêu cầu dữ liệu toàn nhóm còn cần người phụ trách xác minh.
