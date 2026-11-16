# Review công việc M4: truy hồi sản phẩm

Cập nhật ngày 16/11/2026. Tài liệu này đối chiếu code hiện tại với kết quả chạy
đã lưu. [M4_HANDOFF](M4_HANDOFF.md) ghi trạng thái bàn giao ngày 14/11, trước
lần triển khai và đo retrieval dưới đây.

## Những phần đã triển khai

### 1. Nhận ảnh và chuyển mode đến từng crop

Endpoint [`POST /api/search/upload`](../api/main.py) nhận ảnh, `top_k` (mặc định
5, giới hạn 1–20) và `mode` qua query parameter. Ba giá trị hợp lệ là
`histogram_only`, `embedding_only`, `hybrid`; nếu không truyền thì dùng
`hybrid`, còn mode sai trả HTTP 400. API gọi dịch vụ phát hiện quần áo, giải mã
ảnh gốc RGB và truy hồi riêng cho từng crop. Trường hợp không phát hiện quần
áo trả danh sách kết quả rỗng với thông báo từ dịch vụ phát hiện.

Điểm nối M4 nằm ở [`search_match_for_crop`](../api/search_service.py): hàm nhận
mode, crop và ảnh gốc; dùng `crop_variants` do M2 đưa sang nếu có, nếu không
thì dùng ảnh crop chính. API trả `search_mode` trên từng `detected_item`
([`api/schemas.py`](../api/schemas.py)) cùng sản phẩm, điểm, thông báo và lý do
loại. Cơ chế phát hiện và tạo crop là phần đã có của M1/M2; M4 dùng payload đó
để truy hồi.

### 2. Histogram only: tìm theo HSV

1. Lấy descriptor HSV147 của **source ROI** từ ảnh gốc và metadata crop bằng
   `extract_source_color`; kiểm tra config/version/ROI policy của descriptor.
   Mode này không cần chạy FashionCLIP cho ảnh truy vấn.
2. Hàm [`list_color_search_products`](../db/database.py) lấy toàn bộ sản phẩm
   trong PostgreSQL có `embedding_model` và `embedding_version` tương thích.
   Không lấy top ứng viên bằng vector và không lọc cứng theo loại trang phục.
3. Với mỗi ID, lấy HSV catalog từ cache production. Descriptor thiếu/sai bị
   bỏ qua khi `HSV_MISSING_POLICY=zero`; chính sách `error` sẽ báo lỗi. Tính
   `histogram_similarity(query, product)` theo Hellinger, xếp giảm dần; nếu
   bằng điểm thì xếp theo ID. Trả tối đa `top_k` sản phẩm.

Trong mode này `color_score = final_score = similarity`. Tên màu và ô màu
hiển thị được suy xấp xỉ từ histogram; chúng không phải đầu vào tính điểm.
Không áp ngưỡng 0,6. Nếu không có descriptor truy vấn hay ứng viên hợp lệ thì
không có sản phẩm phù hợp. Logic ở [`search_by_histogram`](../api/search_service.py).

### 3. Embedding only: tìm theo vector ảnh

1. Giải mã từng crop variant, dùng FashionCLIP tạo embedding ảnh và kiểm tra
   vector đúng 512 chiều. Mode này không trích HSV truy vấn để xếp hạng.
2. [`search_similar_by_vector`](../db/database.py) tìm trong pgvector bằng
   cosine: `similarity = 1 - (embedding <=> query_vector)`. SQL lọc đúng
   `embedding_model`/`embedding_version`, sắp theo khoảng cách cosine và lấy
   tối đa `max(100, top_k)` ứng viên; không lọc cứng category.
3. Dùng chính cosine similarity làm `image_similarity` và `final_score`.
   Không gọi bộ rerank category/màu/text-style. Màu hiển thị của sản phẩm để
   trống ở nhánh này, tránh thể hiện nhầm màu metadata legacy thành điểm màu.

Khi có nhiều crop variant, kết quả được gộp theo ID sản phẩm và giữ điểm cao
nhất, kèm `best_crop_variant` và `variant_scores`; sau đó chỉ giữ `top_k`.
Đường chạy nằm ở [`_search_single_crop_variant`](../api/search_service.py) và
[`merge_variant_rankings`](../api/search_service.py).

### 4. Hybrid: truy hồi vector rồi rerank

Hybrid dùng cùng truy vấn pgvector và tập ứng viên như embedding only, sau đó
dùng bộ [`rerank_products`](../api/search_service.py) đã có để sắp xếp lại.
Descriptor HSV truy vấn được trích **một lần từ source ROI** và dùng chung cho
các encoder crop variant; descriptor sản phẩm lấy từ cache HSV theo ID. Điểm
cuối đang tính như sau:

```text
base_score = w_ảnh × image_similarity
           + w_loại × category_score
           + w_màu × color_score
           + w_text_style × text_style_score
final_score = max(0, base_score - category_penalty)
```

Trọng số mặc định lần lượt là **0,60 / 0,15 / 0,15 / 0,10**, đọc được từ biến
môi trường `RERANK_*_WEIGHT` và được chuẩn hóa về tổng 1. `image_similarity`
là cosine của vector ảnh; `category_score` so class YOLO với category/title/
brand; `color_score` là độ tương đồng histogram HSV. `text_style_score` kết
hợp độ khớp văn bản sản phẩm và style prompt FashionCLIP theo tỷ lệ 0,65/0,35.
Nhóm loại trang phục đối nghịch có thể bị trừ `category_penalty` (mặc định
0,10). Mã hiện tại không áp penalty màu RGB cũ; khi thiếu HSV, phần màu đóng
góp 0. Nếu text/style không tính được, trọng số của phần đó được phân lại cho
ảnh/loại/màu. Sau rerank, các crop variant được gộp theo điểm tốt nhất cho
từng ID như trên.

### 5. Lọc ngưỡng và dữ liệu trả về

Sau khi gộp variant, **embedding only và hybrid** yêu cầu `final_score > 0,6`.
Điểm đúng 0,6 hoặc thấp hơn bị loại; nếu ứng viên đầu không vượt ngưỡng, crop
được đánh dấu `is_confident_match=false` và không trả sản phẩm gợi ý độ tin
cậy thấp. Nếu ứng viên đầu đạt, từng sản phẩm hiển thị vẫn phải vượt 0,6.
Hai mode này không áp thêm điều kiện `MIN_IMAGE_SIMILARITY` của logic legacy.
Histogram only dùng ngưỡng cơ sở 0, không dùng quy tắc 0,6. Điểm của ba mode
có công thức khác nhau, nên không so trực tiếp như một thang chất lượng chung.

Response mỗi crop có `similar_products`, `top_final_score`,
`match_score_threshold`, `reject_reason`, `search_variants` và `search_mode` để
người dùng/nhóm kiểm tra quyết định. Các thay đổi nằm trong
[`apply_match_threshold`](../api/search_service.py),
[`search_match_for_crop`](../api/search_service.py),
[`api/main.py`](../api/main.py) và [`api/schemas.py`](../api/schemas.py).

### 6. Frontend, Docker và công cụ đánh giá

[`SearchView.jsx`](../web/fe/src/views/SearchView.jsx) thêm ô chọn ba phương
pháp, mặc định hybrid. [`api.js`](../web/fe/src/services/api.js) gửi mode trong
query string của request upload; kết quả tiếp tục hiển thị qua component sản
phẩm hiện có. [`api/Dockerfile`](../api/Dockerfile) cài bản PyTorch CPU trước
dependencies để image API chạy trong môi trường Docker đã kiểm.

[`evaluation/run_eval.py`](../evaluation/run_eval.py) nhận `--mode` và `--limit`,
gửi mode theo từng request, ghi mode vào JSON kết quả và xuất JSON/CSV cho từng
lần chạy. [`labels.retrieval_smoke.json`](../evaluation/labels.retrieval_smoke.json)
gồm 7 ảnh, mỗi nhóm một ảnh; lượt đầy đủ dùng
[`labels.starter_catalog.json`](../evaluation/labels.starter_catalog.json) với
28 ảnh. Mỗi mode có file kết quả riêng trong
[`m4_smoke_results/`](../evaluation/m4_smoke_results/). Docker API, YOLO,
PostgreSQL/pgvector và FashionCLIP đã được dùng trong lượt chạy thực ngày
16/11/2026; đây là bằng chứng của lần chạy đó, không xác nhận trạng thái
container ở thời điểm đọc tài liệu.

## Kiểm thử và kết quả

Chạy trong image `fashion-visual-search-api` với repo mount chỉ đọc và
`--network none`: 4/4 test ở [`test_retrieval_modes.py`](../tests/test_retrieval_modes.py)
đạt; 9/9 test hiện có ở [`test_search_service.py`](../tests/test_search_service.py)
đạt. Test kiểm tra tách biệt nguồn ứng viên histogram/vector, điểm embedding,
ranh giới `> 0,6` và lọc từng sản phẩm khi trả kết quả.

Lượt end-to-end dùng **toàn bộ 28 ảnh** ở `evaluation/queries/starter_catalog/`,
nhãn [`labels.starter_catalog.json`](../evaluation/labels.starter_catalog.json),
`top_k=5` và cùng một catalog 28 sản phẩm cho cả ba chế độ:

| Chế độ | Request thành công | ID ảnh gốc trong top 5 | MRR | Color@5 | Trễ trung bình |
|---|---:|---:|---:|---:|---:|
| Histogram only | 28/28 | 28/28 | 0,982 | 4/28 | 356 ms |
| Embedding only | 28/28 | 28/28 | 0,982 | 0/28 | 703 ms |
| Hybrid | 28/28 | 28/28 | 1,000 | 6/28 | 851 ms |

JSON/CSV từng ảnh và hướng dẫn chạy lại nằm ở
[`RETRIEVAL_MODES_SMOKE.md`](../evaluation/RETRIEVAL_MODES_SMOKE.md),
[`m4_smoke_results/`](../evaluation/m4_smoke_results/). Có thêm lượt 7 ảnh,
mỗi nhóm trang phục một ảnh, dùng
[`labels.retrieval_smoke.json`](../evaluation/labels.retrieval_smoke.json).

## Cách hiểu kết quả và việc còn lại

- Ảnh truy vấn là ảnh sản phẩm **đã nằm trong catalog**. `28/28` đo khả năng tìm
  lại chính ảnh đó, không phải Recall@5 trên ảnh người dùng mới. Nhãn màu trong
  bộ starter được tạo từ metadata, chưa được người kiểm độc lập.
- `Color@5` chỉ so tên `expected_color` với `dominant_color_name` của kết quả;
  nó không phải điểm tương đồng histogram. Cache màu catalog hiện có 28/28 ảnh
  dùng toàn ảnh (`whole_catalog_photo`, `mask_status=missing`), nên nền ảnh có
  thể chi phối màu hiển thị. Cần xác minh nhãn màu và dựng lại cache trên vùng
  quần áo trước khi kết luận độ chính xác màu.
- Chưa có đánh giá retrieval cho ba chế độ trên tập ảnh outfit độc lập có
  `acceptable_product_ids` được người gán. Cần tách validation/test, kiểm
  coverage cache và đo Recall@5/MRR trên tập đó trước khi công bố chất lượng
  tìm kiếm. Ngưỡng 0,6 hiện là quy tắc triển khai, chưa được hiệu chỉnh trên
  tập validation độc lập.
- Thời gian phản hồi ở bảng là số đo tuần tự của lần chạy này, không phải
  benchmark latency được kiểm soát. Lượt 7 ảnh có lần nạp model đầu tiên;
  lượt 28 ảnh chạy sau khi dịch vụ đã được dùng.

Đối chiếu yêu cầu bàn giao còn mở tại [M4_HANDOFF](M4_HANDOFF.md), đặc biệt
cache catalog production theo ROI phù hợp và đánh giá trên ảnh truy vấn độc lập.
