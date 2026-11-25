# Benchmark Fashion Visual Search

Cập nhật ngày 17/11/2026. Benchmark đo khả năng tìm sản phẩm cho **một món mục tiêu trên mỗi ảnh**. Nhãn bbox, loại, màu và ánh sáng của ảnh truy vấn do người gán. **Nhãn độ phù hợp của sản phẩm (đúng loại, đúng màu, chấp nhận được) do AI (Claude) chấm theo tiêu chí cố định; người dùng đã rà lại 25/75 ảnh (209 thẻ), đồng ý với AI 90,9% (kappa 0,82)**; xem mục Giới hạn. Retrieval dùng cosine chính xác trên catalog xuất ra JSON, chạy trong một tiến trình trên CPU.

## Dữ liệu và trạng thái

| Tập | Nguồn | Số ảnh hiện có | Trạng thái |
|---|---|---:|---|
| Validation | Ảnh Wikimedia Commons trong `real_outfits` có bbox từ bộ M1, cộng 8 ảnh mới | 23 (tối thiểu 20) | Đã chấm, đã khóa cấu hình |
| Test | 5 ảnh `m1_*` từ bộ M1 và 47 ảnh Commons mới từ danh sách ứng viên | 52 (tối thiểu 50) | Đã chạy một lần sau khi khóa, đã chấm |
| Ứng viên | Ảnh Wikimedia Commons với nguồn, tác giả và giấy phép trong [candidates.csv](queries/bench/candidates.csv) | 163 | Đã chọn và gán nhãn 75 ảnh |
| Robustness | 28 ảnh từ `starter_catalog` và 10 ảnh từ `auto_smoke`, mỗi ID qua 7 điều kiện | 266 | Đã đo |

Hai manifest ghi nguồn và giấy phép từng ảnh. Các ảnh `real_outfits` cũ chỉ được dùng cho validation. Đối chiếu các tập theo ID và SHA-256 ảnh. Công cụ `manifest check` cảnh báo khi một tác giả xuất hiện ở cả hai tập. Lần kiểm tra đầu phát hiện 3 ảnh test cùng tác giả hoặc cùng sự kiện với ảnh validation (`bench_shorts_01`, `bench_black_skirt_01`, `bench_red_dress_03`); cả ba được chuyển sang validation trước khi chạy bất kỳ kết quả nào, nên hiện không còn cảnh báo.

| Tập | Ánh sáng | Loại | Màu |
|---|---|---|---|
| [Validation](bench_validation.csv) | dark 6, normal 17 | long_sleeve_top 1, short_sleeve_dress 3, short_sleeve_top 8, shorts 3, skirt 3, trousers 4, vest_dress 1 | black 9, blue 2, brown 2, gray 1, green 3, navy 1, olive 1, pink 1, red 1, white 2 |
| [Test](bench_test.csv) | dark 4, normal 48 | long_sleeve_dress 4, long_sleeve_outwear 8, long_sleeve_top 9, short_sleeve_dress 1, short_sleeve_top 12, shorts 4, skirt 2, trousers 5, vest 5, vest_dress 2 | beige 1, black 13, blue 7, brown 3, cream 1, gray 2, navy 6, olive 3, orange 1, pink 1, red 1, white 12, yellow 1 |

Validation có 22 ảnh `has_match` và 1 ảnh `no_match`. Test có 47 `has_match`, 4 `no_match` và 1 ảnh chưa xác nhận (`bench_shorts_07`: sau khi người dùng rà, không còn sản phẩm nào được chấp nhận và chưa xem danh sách mở rộng để kết luận `no_match`), nên bị loại khỏi mẫu số. Đánh dấu ba thuộc tính ở cấp sản phẩm gồm đúng loại, đúng màu chủ đạo và chấp nhận được. Một sản phẩm chấp nhận được phải đúng loại, màu chủ đạo và đủ gần về kiểu dáng hoặc họa tiết. Trang chấm trộn top-10 từ các cấu hình và top-20 cosine trên ROI nhãn, xáo thẻ theo query ID và ẩn tên cấu hình, điểm và hạng. Xác nhận `no_match` sau khi xem phần mở rộng gồm tối đa 60 sản phẩm có điểm loại dương, sắp bằng cosine trước khi xáo. Khi đã có sản phẩm chấp nhận được, trạng thái là `has_match`. Các ca chưa xác nhận bị loại khỏi mẫu số chất lượng.

## Tám tiêu chí

| Tiêu chí | Định nghĩa |
|---|---|
| Phát hiện mục tiêu | Chọn detection đúng lớp có IoU cao nhất với bbox nhãn sau EXIF và resize. Ghép nếu IoU ≥ 0,5. Recall chia theo mọi truy vấn, không phụ thuộc việc chấm sản phẩm, có kết quả riêng theo ánh sáng. A0 bỏ qua metric này. |
| Retrieval trước ngưỡng | HitRate@5 đo có sản phẩm được chấp nhận trong 5 hạng đầu. MRR@5 dùng nghịch đảo hạng đầu tiên đạt điều kiện. Precision@5 chia số sản phẩm chấp nhận được trong top-5 cho 5. Mẫu số trung bình là các truy vấn `has_match`. Miss detection tính 0. |
| Joint loại+màu@5 | Cần một sản phẩm trong top-5 đúng cả loại và màu. Chia theo số truy vấn đã xác nhận. |
| Sau quyết định | Truy vấn bị từ chối có HitRate@5 và RR@5 bằng 0. Coverage là tỷ lệ ghép được và được chấp nhận. False acceptance chỉ xét `no_match`. Báo số đếm nếu có dưới 5 ca `no_match`. |
| Ứng viên | Candidate hit@100 xét hợp các top-100 của từng crop. Mẫu số là `has_match`. |
| Màu truy vấn | So tên màu từ crop tight với nhãn người trên các mục tiêu ghép được, kèm ma trận nhầm lẫn. |
| Thời gian | Báo mean và p95 theo công đoạn và tổng trên các ảnh sau ảnh đầu. Báo riêng ảnh đầu. Công đoạn dùng chung giữ thời gian đo lần tính thật. |
| So sánh theo cặp và lỗi | So RR@5 trước ngưỡng của từng `has_match` với B0. Gán lỗi theo thứ tự `no_detection`, `wrong_class`, `candidate_miss`, `rank_miss`, `rejected`, `hit`. |

HitRate là phép đo có kết quả phù hợp trong top-k. Recall cần tổng số sản phẩm phù hợp trong catalog. Candidate hit@100 là cận dưới vì nhãn độ phù hợp chỉ đầy đủ trong phạm vi đã chấm. Sản phẩm top-5 chưa chấm được tính 0 và cộng vào `unjudged_at_5`. Bảng cuối cần đủ chấm và `unjudged_at_5 = 0`.

## Cấu hình so sánh

| Cấu hình | Tham số và lý do | Kết quả outfit |
|---|---|---|
| A0_whole_image | FashionCLIP trên ảnh đã chuẩn hóa, cosine thuần. Cô lập tác dụng của ROI. Luôn chấp nhận danh sách. | Đã đo |
| C1p_color_only | Đối chứng cổ điển dùng 0,50 điểm màu trội và 0,50 điểm từ khóa loại trên toàn catalog. Dùng trọng số bằng nhau để có mốc đơn giản. | Đã đo |
| C2_embedding_only | ROI YOLO, trọng số ảnh 1, tắt text/style cùng phạt màu và loại đối nghịch. Đo cosine từ ROI. | Đã đo |
| B0_current | Pipeline gốc với trọng số ảnh/loại/màu/text-style 0,60/0,15/0,15/0,10. Dùng làm mốc ghép cặp. **Trong benchmark in-process, B0 chạy không có descriptor màu truy vấn nên điểm màu luôn bằng 0** (hybrid thật, qua API, có truyền descriptor). | Đã đo |
| B1_gamma | B0 và gamma 0,70 trên nhánh YOLO khi mean luma < 0,30. Lấy cấu hình M1 đã khóa. | Đã đo |
| C3_w1 | 0,70/0,10/0,10/0,10. Thử tăng vai trò ảnh trên validation. Được chọn làm C3_selected. | Đã đo |
| C3_w2 | 0,50/0,20/0,20/0,10. Thử tăng vai trò loại và màu trên validation. | Đã đo (chỉ validation) |
| C4_group_terms | B0 với ba lớp `long_sleeve_top`, `long_sleeve_outwear`, `vest` dùng chung bộ từ khóa loại (cờ `CATEGORY_GROUP_UPPER`). | Chỉ validation |
| C5_hsv_color / C5_hsv_color_w1 | B0 / C3_w1 cộng descriptor HSV147 của ROI truy vấn (đúng như API). | Validation, thăm dò test |
| C6_center, C6_intersection, C6_center_intersection | C5 với ROI trung tâm 60%, metric intersection thay Hellinger, hoặc cả hai. | Validation, thăm dò test |
| B2 / B3 / B4 | Mask thô / mask và morphology / cân bằng trắng và HSV | Chưa đo |
| C1 | Đối chứng histogram HSV của nhóm | Chưa đo |
| F | Pipeline tích hợp cuối | Chưa đo |

Dùng YOLO `conf=0.5` và crop tight/padded/wide mở rộng 0%/15%/30%, pad trắng và đưa về 224×224 để khớp production. Mỗi crop đi qua vòng JPEG trước khi trích đặc trưng. B0 và các biến thể hybrid giữ ngưỡng phạt màu 0,35, mức phạt màu 0,08, phạt loại đối nghịch 0,10, ngưỡng điểm cuối 0,70 và ngưỡng tương đồng ảnh 0,60 theo production. Truy xuất top-100 theo từng crop, rerank tối đa 20 mỗi crop và gộp top-10 theo điểm cao nhất của từng ID. B1 chỉ thay tiền xử lý cho YOLO. C1 của nhóm dùng histogram HSV, còn C1p dùng màu trội và từ khóa.

Validation chạy tất cả cấu hình hiện có. Chọn cấu hình có MRR@5 trước ngưỡng cao nhất trong B0/C3_w1/C3_w2, ưu tiên B0 khi hòa. Khóa lưu hàm băm model, catalog và manifest validation, hash ảnh, ngưỡng YOLO, crop và định nghĩa cấu hình. Thư mục test được giữ chỗ trước suy luận. Kết quả từng truy vấn được ghi ngay; nếu lượt chạy bị ngắt, chạy lại cùng lệnh với `--resume`. Truy vấn lỗi suy luận được giữ lại và tính là trượt. Văn bản catalog được mã hóa sẵn trước khi đo để latency rerank của các cấu hình so được với nhau. Chạy test một lần với A0, C1p, C2, B0, B1 và thêm C3_selected nếu khác B0. Tính lại điểm từ inference lưu sẵn và nhãn chấm.

## Kết quả đã chạy

**Smoke.** Ba ảnh validation chạy qua đủ 7 cấu hình, mỗi mục tiêu ghép được có 10 sản phẩm, kèm kết quả IoU và thời gian. [Kiểm tra smoke](../docs/benchmark/smoke/checks.csv)

**So B0 với API cũ.** So theo đúng lớp và IoU trong hệ tọa độ của mỗi lượt. Số top-5 trùng đủ ID và thứ tự là **0/3**. Cả 15 ID top-5 cũ đều nằm trong hợp ứng viên hiện tại. Ảnh áo xanh giữ bbox và màu truy vấn; điểm màu đổi do `color_similarity` áp trần cho các cặp tên màu lệch, đồng thời điểm text/style có chênh lệch. Ảnh quần đen đi từ 960×1442 sang 852×1280 ở M1, nên vị trí crop và độ tương đồng có thể đổi; các điểm màu lệch tên cũng giảm. Ảnh đầm giữ bốn hạng đầu. API cũ lưu top-5 nhưng thiếu tập ứng viên và phiên bản thư viện, nên chưa xác định được nguyên nhân đổi hạng năm. [Đối chiếu top-5](../docs/benchmark/smoke/legacy_top5.csv)

**Robustness.** Dùng đúng product ID, lấy hạng tốt nhất trong mọi detection trước ngưỡng. Original giữ JPEG nguồn. Điều kiện biến đổi lưu JPEG chất lượng 95 trừ `jpeg_q10`. Dùng seed 20261116, làm tối bằng cách nhân pixel với 0,25, downsample BOX còn 25% mỗi chiều, JPEG quality 10, Gauss σ=20, xoay 15° nền trắng và cửa sổ crop 75% dịch 10 điểm phần trăm từ tâm về góc trên trái. Hai lượt build cho cùng SHA-256 trên 266 ảnh.

**Bảng dưới đo trước khi đổi điều kiện `dark`.** Bản cũ làm tối bằng gamma 2,5 nên nền trắng giữ nguyên và gamma của M1 chỉ bật ở 7/266 ảnh; vì vậy B1 gần như lặp lại B0. Ảnh `dark` đã được dựng lại, **chưa đo lại**: chạy `run robustness` với thư mục kết quả mới rồi thay bảng này và các tệp trong `docs/benchmark/robustness/`.

| Điều kiện | A0 HitRate@5 | B0 HitRate@5 | B1 HitRate@5 | B0/B1 không detection |
|---|---:|---:|---:|---:|
| original | 0,974 (37/38) | 0,947 (36/38) | 0,947 (36/38) | 0,000 (0/38) |
| dark | 0,947 (36/38) | 0,921 (35/38) | 0,921 (35/38) | 0,000 (0/38) |
| downsample | 0,974 (37/38) | 0,737 (28/38) | 0,737 (28/38) | 0,237 (9/38) |
| jpeg_q10 | 0,974 (37/38) | 0,684 (26/38) | 0,684 (26/38) | 0,237 (9/38) |
| gauss_noise | 0,974 (37/38) | 0,921 (35/38) | 0,921 (35/38) | 0,026 (1/38) |
| rotate15 | 1,000 (38/38) | 0,947 (36/38) | 0,947 (36/38) | 0,026 (1/38) |
| offcrop | 0,974 (37/38) | 0,868 (33/38) | 0,868 (33/38) | 0,026 (1/38) |

B0 và B1 có cùng kết quả tổng hợp trong cả bảy điều kiện. JPEG quality 10 gây mức giảm HitRate@5 lớn nhất ở hai cấu hình ROI, từ 36/38 xuống 26/38. Downsample và JPEG quality 10 đều có 9/38 ảnh không tạo detection. Chương 2 liên quan resize và xoay, chương 3 liên quan gamma và nhiễu, chương 4 liên quan màu, chương 6 liên quan nén JPEG. [Bảng đầy đủ](../docs/benchmark/robustness/summary.md) · [Từng truy vấn](../docs/benchmark/robustness/per_query.csv) · [Môi trường và hash](../docs/benchmark/robustness/run_metadata.csv)

## Kết quả validation và test

Chạy ngày 17/11/2026. Cấu hình khóa trên validation, sau đó test chạy **một lần**. Tập test không dùng để chọn gamma, trọng số, ngưỡng hay `conf`. Khóa gồm `conf=0,5`, crop tight/padded/wide và C3_selected = C3_w1 (MRR@5 trước ngưỡng cao nhất). Bảng dưới dùng nhãn mới nhất, gồm 25 ảnh đã được người rà lại. [Khóa cấu hình](../docs/benchmark/main/locked_config.json) · [Ngưỡng đã khóa](../docs/benchmark/main/locked_thresholds.json) · [Nhãn độ phù hợp](../docs/benchmark/main/bench_qrels.csv)

### Tóm tắt các thông số (cấu hình tốt nhất B0 hoặc C3_selected)

| Thông số | Test | Validation |
|---|---|---|
| Phát hiện mục tiêu, đúng lớp, IoU ≥ 0,5 (recall) | 0,519 (27/52); ảnh tối 3/4 | 0,739 (17/23); ảnh tối 4/6 |
| Phát hiện, chỉ xét vị trí (bỏ điều kiện lớp, 75 ảnh gộp) | 0,83 (62/75) | |
| HitRate@5 (chỉ ảnh `has_match`) | **0,426** (20/47) | 0,500 (11/22) |
| MRR@5 | 0,363 (B0), 0,355 (C3_selected) | 0,305 (B0), 0,339 (C3_w1) |
| Precision@5 | 0,221 (52/235) | 0,218 (24/110) |
| Joint loại+màu@5 | 0,412 (21/51) | 0,478 (11/23) |
| Candidate hit@100 (cận dưới) | 0,511 (24/47) | 0,727 (16/22) |
| Sau ngưỡng đã hiệu chỉnh (T 0,45, M 0,40): HitRate@5 | 0,426 | 0,500 |
| Sau ngưỡng: độ chính xác khi chấp nhận / coverage | 0,80 / 0,52 | 0,75 / 0,74 |
| Chấp nhận nhầm `no_match` | 1/4 | 1/1 |
| Sau ngưỡng production 0,70: HitRate@5 / coverage | 0,000 / 0,000 | 0,000 / 0,000 |
| Màu truy vấn khớp nhãn người | 0,222 (6/27) | 0,353 (6/17) |
| Thời gian CPU sau ảnh đầu (mean / p95, ms) | B0 1417 / 3317; C2 742 / 1596; A0 80 / 109 | B0 2033 / 4216 |
| Mask IoU | chưa đo | chưa đo |

### Retrieval trước ngưỡng

Đo trên ảnh `has_match`; miss detection tính 0; ghép mục tiêu đúng lớp, IoU ≥ 0,5. HitRate@5 gộp lớp là phân tích bổ sung (xem bên dưới).

| Cấu hình | Validation HitRate@5 | Validation MRR@5 | Test HitRate@5 | Test MRR@5 | Test Precision@5 | Test HitRate@5 gộp lớp |
|---|---:|---:|---:|---:|---:|---:|
| A0_whole_image | 0,227 (5/22) | 0,114 | 0,340 (16/47) | 0,214 | 0,111 | 0,340 |
| C1p_color_only (đối chứng cổ điển) | 0,227 (5/22) | 0,121 | 0,213 (10/47) | 0,128 | 0,064 | 0,277 |
| C2_embedding_only | 0,455 (10/22) | 0,311 | 0,362 (17/47) | 0,266 | 0,187 | 0,468 |
| B0_current | 0,500 (11/22) | 0,305 | **0,426 (20/47)** | **0,363** | 0,221 | 0,553 |
| B1_gamma | 0,500 (11/22) | 0,305 | 0,426 (20/47) | 0,363 | 0,217 | 0,553 |
| C3_w1 / C3_selected | 0,500 (11/22) | 0,339 | 0,426 (20/47) | 0,355 | 0,230 | **0,574** |
| C3_w2 | 0,455 (10/22) | 0,295 | không chạy test | | | |

Hybrid (B0, C3) cao hơn embedding-only và đối chứng màu trên cả hai tập: ở test 20 ảnh so với 17 (C2) và 10 (C1p). Chênh lệch B0 và C2 chỉ 3 ảnh, nên chỉ nói "có xu hướng tốt hơn". A0 (không detection) thấp hơn B0 (16 so với 20 ảnh), nên detection và ROI có đóng góp nhưng chỉ khi tìm đúng món. Chênh lệch 0 đến 1 ảnh giữa B0, B1 và C3 không đủ để kết luận. [Tóm tắt validation](../docs/benchmark/main/validation/summary.md) · [Tóm tắt test](../docs/benchmark/main/test/summary.md) (hai file tóm tắt in ngưỡng production 0,70 trong bảng "Sau quyết định"; ngưỡng đã hiệu chỉnh ở mục sau)

### Ngưỡng chấp nhận

Ngưỡng production 0,70 (điểm cuối) loại gần như mọi kết quả: HitRate@5 sau quyết định của B0 bằng 0 trên cả hai tập. Quét lưới T ∈ [0,30; 0,70] và ảnh tối thiểu M ∈ {không, 0,3 đến 0,6} trên **validation**; quy tắc chọn cố định trước: tối đa hóa HitRate@5 sau quyết định, nếu hòa thì T cao nhất, rồi M cao nhất. Kết quả: **T = 0,45, M = 0,40** cho B0 và C3, T = 0,50, M = 0,50 cho C2. Chạy lại phép quét với nhãn mới (sau khi người rà) cho cùng cặp tham số. Test chỉ áp lại cặp đã khóa.

| Cấu hình | Tập | HitRate@5 sau quyết định | Độ chính xác khi chấp nhận | Coverage | Chấp nhận nhầm `no_match` |
|---|---|---:|---:|---:|---:|
| B0, T=0,45 M=0,40 | validation | 0,500 | 0,75 | 0,74 | 1/1 |
| B0, T=0,45 M=0,40 | test | 0,426 | 0,80 | 0,52 | 1/4 |
| C2, T=0,50 M=0,50 | validation | 0,455 | 0,73 | 0,70 | 1/1 |
| C2, T=0,50 M=0,50 | test | 0,362 | 0,71 | 0,50 | 1/4 |

Không có ngưỡng nào vừa giữ coverage vừa chặn ca `no_match`: ca `no_match` của validation có điểm 0,64 đến 0,73, cao hơn nhiều ca đúng. Chỉ có 1 ca (validation) và 4 ca (test) `no_match`, nên chưa kết luận được về false acceptance. [Bảng quét](../docs/benchmark/main/validation/threshold_sweep.csv)

### Phát hiện mục tiêu là nút thắt lớn nhất

Trên 75 ảnh, 44 ảnh (0,587) ghép được đúng lớp; test chỉ đạt 27/52 (0,519). Trong 31 ảnh trượt, 18 ảnh YOLO tìm đúng vị trí nhưng gán sai loại (IoU ≥ 0,5), 13 ảnh không có khung nào chồng đủ. Nếu bỏ điều kiện đúng lớp thì recall vị trí là 62/75 (0,83). Nhầm lẫn lớn nhất nằm ở nhóm áo khoác: `long_sleeve_outwear` thành `long_sleeve_top` (4), `vest` thành `long_sleeve_top` hoặc `long_sleeve_outwear` (4). Khi chấm, nhiều ảnh gán `vest` có món nhìn thấy là blazer hoặc áo khoác, nên một phần lỗi do ranh giới lớp mờ. Ghép độc lập bằng `ultralytics` cho kết quả trùng 44/75, xác nhận phép tính. Về nguyên nhân thất bại end-to-end ở test (B0, 52 ảnh): 8 ảnh không phát hiện, 14 ảnh sai lớp, 1 ảnh thiếu ứng viên, 6 ảnh xếp hạng chưa đưa sản phẩm đúng vào top-5.

**Ngưỡng `conf` của YOLO.** Thử `conf=0,25` trên **validation** (lượt riêng, không đụng test): recall mục tiêu vẫn 17/23, HitRate@5 và MRR@5 của B0 không đổi; một ảnh đổi từ `no_detection` sang `wrong_class`. Giữ `conf=0,5`. [Kết quả](../evaluation/results/bench_run3_conf025/validation/summary.md)

### Gộp lớp mờ (phân tích bổ sung, không thay thế bảng chính)

Gộp `long_sleeve_top`, `long_sleeve_outwear` và `vest` thành một nhóm, cố định trước khi xem kết quả, rồi tính lại từ inference đã lưu bằng `python -m evaluation.benchmark.class_groups`. Mọi sản phẩm top-5 đều đã được chấm. Đây là định nghĩa đánh giá khác, nên luôn báo cùng bản nghiêm ngặt ở trên.

| Cấu hình | Tập | Ảnh phát hiện: nghiêm ngặt → gộp | HitRate@5: nghiêm ngặt → gộp | MRR@5: nghiêm ngặt → gộp |
|---|---|---:|---:|---:|
| B0_current | validation | 16 → 17 | 0,500 → 0,545 | 0,305 → 0,327 |
| C3_w1 | validation | 16 → 17 | 0,500 → 0,545 | 0,339 → 0,384 |
| B0_current | test | 25 → 33 | 0,426 → 0,553 | 0,363 → 0,434 |
| C2_embedding_only | test | 25 → 33 | 0,362 → 0,468 | 0,266 → 0,315 |
| C3_selected | test | 25 → 33 | 0,426 → 0,574 | 0,355 → 0,430 |

Thứ hạng giữa các cấu hình không đổi sau khi gộp.

### Các hướng cải tiến đã thử (ablation)

Đo trên validation, quy trình giống nhau, nhãn chấm bổ sung cho sản phẩm mới ở top-5. Không hướng nào vượt B0 một cách đáng tin.

| Hướng | Cấu hình | Validation HitRate@5 | Kết luận |
|---|---|---:|---|
| Nhóm từ khóa loại | C4_group_terms | 0,500 = B0 | Bằng B0; cờ đổi top-5 của 10/83 detection nhưng không chuyển ảnh nào từ trượt sang trúng |
| Thêm màu HSV truy vấn | C5_hsv_color | 0,409 (9/22) | Kém B0 |
| Màu HSV, ROI trung tâm | C6_center | 0,500 (11/22) | Bằng B0 |
| Màu HSV, intersection | C6_intersection | 0,409 (9/22) | Kém B0 |
| Màu HSV, trung tâm + intersection | C6_center_intersection | 0,500 (11/22), MRR cao hơn 0,036 | Thắng 5 thua 4, trong nhiễu |

**Phát hiện về B0.** Điểm màu của B0 trong benchmark luôn bằng 0 (pipeline in-process không truyền descriptor màu HSV truy vấn). Điểm màu HSV thật có sức phân biệt yếu: AUC 0,605 so với nhãn phù hợp và 0,544 so với nhãn màu của người, trong khi điểm text/style có AUC 0,82 và điểm loại 0,66.

**Thăm dò trên test (sau khi đã thấy test, không độc lập).** Chạy C5 và C6 trên test bằng `explore_test.py`, thư mục riêng, không đụng khóa cấu hình. HitRate@5 test (47 ảnh): B0 20, C5 21, C6_center 22, C6_intersection 21, C6_center_intersection 22; MRR@5 0,363 / 0,362 / 0,362 / 0,332 / 0,384. So với B0, C6_center_intersection thắng 4, thua 3, hòa 40 (kiểm định dấu p = 1,00). Kết luận: thêm điểm màu HSV và ROI trung tâm không tạo cải thiện có ý nghĩa thống kê, nên cấu hình cuối vẫn là B0/C3_selected. [Kết quả thăm dò](../evaluation/results/bench_run7_explore_test/results.json)

### Độ tin cậy của nhãn sản phẩm

Người dùng rà lại 25 ảnh (20 test, 5 validation), 209 thẻ gồm sản phẩm AI chấp nhận và top-5 của B0, C2, C3. Đồng ý ở nhãn "chấp nhận": 190/209 (90,9%), kappa 0,82; đúng loại 95,7%, đúng màu 92,3%. Người dùng sửa 19 nhãn "chấp nhận" (13 thẻ AI từ chối mà người chấp nhận, 6 thẻ AI chấp nhận mà người từ chối). Nhãn của 25 ảnh này được thay bằng nhãn người (người chấm ghi trong dữ liệu); 50 ảnh còn lại vẫn là nhãn AI. [Nhãn người rà](../docs/benchmark/main/grades_review_human.json)

### Khác

**Gamma M1.** B1_gamma bằng B0 ở HitRate@5 trên cả hai tập. Test chỉ có 4 ảnh tối (3/4 phát hiện được) nên không đủ để kết luận lợi ích của gamma trên ảnh thiếu sáng.

**Màu truy vấn.** Tên màu từ crop tight khớp nhãn người ở 6/17 (0,353) trên validation và 6/27 (0,222) trên test. Màu đen hay bị gán thành `navy`.

**Thời gian** (CPU, sau ảnh đầu, mean/p95 ms, test): A0 tổng 80/109; C2 tổng 742/1596; B0 tổng 1417/3317, trong đó rerank 688/1820. Ảnh đầu của B0 mất 4123 ms vì nạp model. Rerank text/style là phần chậm nhất của B0.

**Chưa đo.** B2, B3, B4, C1 (histogram HSV qua API), F và Mask IoU chưa chạy: runner in-process chưa có các cấu hình này và mask mặc định tắt. C1p là đối chứng cổ điển duy nhất trong bảng.

## Chạy lại từ gốc repo

Dùng `venv` hiện có. Nếu chuyển máy, lấy bản catalog đủ 1.752 sản phẩm và checkpoint YOLO từ chủ dự án, đặt tại `data/fashion_with_vectors.json` và `models/object-detection/best.pt`. Đối chiếu SHA-256 trong [run_metadata.csv](../docs/benchmark/robustness/run_metadata.csv). Lấy thêm `evaluation/labels.auto_smoke.json` và thư mục `evaluation/queries/auto_smoke/` từ máy dự án nếu thiếu. Lần đầu FashionCLIP tải weights vào Hugging Face cache.

```powershell
$python = ".\venv\Scripts\python.exe"
$run = "evaluation/results/bench_reproduce"
& $python -m pip install --dry-run -r requirements-benchmark.txt
& $python -m pip install -r requirements-benchmark.txt
& $python -m unittest discover -s tests -v
& $python -m evaluation.benchmark.manifest check
& $python -m evaluation.benchmark.run validation --run-dir $run --smoke 3
& $python -m evaluation.benchmark.build_perturbed
& $python -m evaluation.benchmark.run robustness --run-dir $run
```

Giữ CPU wheel `torch 2.11.0+cpu` và `torchvision 0.26.0+cpu`. Đọc kết quả dry-run trước khi cài. Chỉ cài `opencv-python` mà Ultralytics yêu cầu; không cài thêm bản headless vì hai wheel ghi đè tệp `cv2` của nhau. Venv đã dùng Transformers 4.57.6 vì 5.4.0 trả cấu trúc khác cho hàm trích đặc trưng hiện tại. Khi đã có cache tại `.hf_cache`, có thể đặt `HF_HOME` vào thư mục đó và `HF_HUB_OFFLINE=1`. Lượt đo đã đặt `OMP_NUM_THREADS=4` và `MKL_NUM_THREADS=4`. Chỉ chạy `manifest init` khi chưa có hai manifest. `build_perturbed` cần `--force` nếu chủ động dựng lại ảnh đã có.

### 1. Gán nhãn ảnh

Nếu thiếu bộ ứng viên, chạy `& $python -m evaluation.benchmark.fetch_candidates --per-category 10 --width 900`. Công cụ tra danh mục và metadata qua [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Categorymembers).

Mở [label_tool.html](benchmark/label_tool.html) bằng Chrome hoặc Edge. Chọn `evaluation/queries/bench/candidates.csv` và thư mục `evaluation/queries/bench/`. Chọn đúng một món mỗi ảnh, vẽ bbox, chọn loại, màu, ánh sáng và tập. Xuất CSV khi có thêm 5 ảnh validation và ít nhất 45 ảnh test. Ước lượng khoảng 1–2 giờ. Trang chấm cần các biến `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` và `R2_BUCKET_NAME` để tải ảnh sản phẩm. Chúng đã được đặt trên máy đang chạy benchmark. Quyền tải ảnh sẽ được xác minh ở bước tạo trang chấm.

```powershell
& $python -m evaluation.benchmark.manifest add "$env:USERPROFILE/Downloads/bench_labeled.csv"
& $python -m evaluation.benchmark.manifest check
& $python -m evaluation.benchmark.run validation --run-dir $run
& $python -m evaluation.benchmark.grade pool --run-dir $run --stage validation
```

### 2. Chấm validation và khóa

Mở `$run/validation/grading.html`, chấm các thẻ, bấm "Đánh dấu đã chấm xong" cho từng truy vấn rồi xuất `grades_validation.json`. Chỉ truy vấn đã đánh dấu mới được xuất; thẻ của truy vấn chưa đánh dấu vẫn là chưa chấm. Với `--no-images`, dùng link sản phẩm để xem ảnh. Thời lượng chấm phụ thuộc số thẻ trong pool. Hoàn thành cả top-5 của mọi cấu hình trước khi khóa.

```powershell
& $python -m evaluation.benchmark.grade merge "$env:USERPROFILE/Downloads/grades_validation.json"
& $python -m evaluation.benchmark.score --run-dir $run --stage validation
& $python -m evaluation.benchmark.run test --run-dir $run
& $python -m evaluation.benchmark.grade pool --run-dir $run --stage test
```

### 3. Chấm test và lưu báo cáo

Mở `$run/test/grading.html`, chấm rồi xuất `grades_test.json`.

```powershell
& $python -m evaluation.benchmark.grade merge "$env:USERPROFILE/Downloads/grades_test.json"
& $python -m evaluation.benchmark.score --run-dir $run --stage test
```

Hiệu chỉnh ngưỡng trên validation rồi áp lại cặp đã khóa lên test, và tính lại với lớp gộp:

```powershell
& $python -m evaluation.benchmark.threshold fit --run-dir $run
& $python -m evaluation.benchmark.threshold apply --run-dir $run
& $python -m evaluation.benchmark.class_groups --run-dir $run --stage test
```

Đặt `HSV_COLOR_CACHE_PATH` trỏ tới cache màu đủ 1.752 sản phẩm trước khi chạy `run`, nếu không điểm màu của hầu hết sản phẩm bằng 0.

Chép `locked_config.json`, các `summary.md`, `summary.csv`, `per_query.csv` và thư mục `examples/` có ảnh và giấy phép từ lượt đo vào `docs/benchmark/`. Đo B2–B4, C1 hoặc F qua API bằng một lượt riêng có `--api-url http://localhost:8000 --config-name F` cho validation và test. Dùng cùng URL và tên sau khi khóa. Adapter báo `n/a` cho candidate hit và thời gian từng công đoạn khi API thiếu các trường đó.

## Giới hạn

Cỡ mẫu nhỏ (23 ảnh validation, 52 ảnh test) và mỗi ảnh chỉ đánh giá một món. **Nhãn độ phù hợp của sản phẩm do AI (Claude) chấm; chỉ 25/75 ảnh được người rà lại (đồng ý 90,9%, kappa 0,82), 50 ảnh còn lại là nhãn AI chưa kiểm.** Tiêu chí cố định: loại theo món nhìn thấy trong ảnh (không theo nhãn lớp khi hai thứ lệch nhau), màu chủ đạo khớp nhãn người, kiểu dáng đủ gần; cardigan và áo len không tính là blazer. Người chấm chỉ xem ảnh đã xáo thứ tự, không thấy hạng hay cấu hình. Kế hoạch (trang 10) yêu cầu nhãn do người gán và kiểm chéo, nên các số liệu trên cần người rà lại phần còn lại trước khi dùng làm kết luận chính; dữ liệu ghi `grader` rõ là AI hoặc tên người rà. Nhãn bbox, loại, màu và ánh sáng của ảnh truy vấn do người gán. Một số ảnh có nhãn lớp lệch món nhìn thấy (ví dụ nhãn `vest` nhưng là blazer). Nhãn phù hợp chỉ đầy đủ trên pool đã xem; 5 ca `no_match` (1 validation, 4 test) là mẫu quá nhỏ cho false acceptance. Số đo in-process dùng cosine chính xác và CPU không qua HTTP; pgvector có thể dùng chỉ mục xấp xỉ. Tập ảnh biến đổi đánh giá tự truy xuất với chính product ID trong catalog, nên diễn giải mức sụt so với original. Kết luận về M2–M4 cần kết quả đo sau tích hợp. Ảnh test có chỉ 4 ảnh thiếu sáng, nên kết luận về M1 gamma cần bộ ảnh tối lớn hơn. Ngưỡng 0,45/0,40 chọn trên 22 ảnh validation nên chưa chắc ổn định. Mask (M2) mặc định tắt trong lượt đo này và cache màu catalog là ROI không mask (`mask_status=missing`).

## Bộ hồi quy cũ

`run_eval.py` và `run_benchmark.py` vẫn dùng để kiểm tra hồi quy API. Điểm gộp 0,8152 của bộ cũ là điểm tổng hợp tự đặt. Accuracy cần nhãn và mẫu số theo metric cụ thể. Các số cũ dùng giao thức khác nên không đưa vào bảng benchmark mới.

**Công cụ AI** · OpenAI Codex hỗ trợ viết công cụ và test, chạy đo, kiểm tra xuất tệp trên Chromium và soạn tài liệu. Claude (Anthropic, `claude-sonnet-5-5`) chạy lượt validation và test, hiệu chỉnh ngưỡng, thêm `threshold.py` và `class_groups.py`, **chấm độ phù hợp của sản phẩm** (khoảng 2.700 cặp; xem `grader` trong [dữ liệu chấm](../docs/benchmark/main/)), thử các cải tiến C4 đến C6 và cập nhật tài liệu này; 25 ảnh được người dùng rà lại. Nhãn bbox, loại, màu và ánh sáng của ảnh truy vấn do người dùng gán qua công cụ gán nhãn.
