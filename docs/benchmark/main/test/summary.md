CHƯA HOÀN TẤT CHẤM

Truy vấn unconfirmed · bench_shorts_07

# Benchmark test

## Tập dữ liệu

- 52 ảnh. has_match 47, no_match 4, unconfirmed 1.
- Lớp · long_sleeve_dress 4, long_sleeve_outwear 8, long_sleeve_top 9, short_sleeve_dress 1, short_sleeve_top 12, shorts 4, skirt 2, trousers 5, vest 5, vest_dress 2.
- Ánh sáng · dark 4, normal 48.
- Màu · beige 1, black 13, blue 7, brown 3, cream 1, gray 2, navy 6, olive 3, orange 1, pink 1, red 1, white 12, yellow 1.

## Trước ngưỡng

C1p_color_only là đối chứng màu trội và từ khóa. C1 của nhóm dùng histogram HSV.
HitRate chỉ đo có sản phẩm chấp nhận được trong top-k. Recall cần biết toàn bộ sản phẩm phù hợp. Candidate hit@100 là cận dưới trong phạm vi đã chấm.

| Cấu hình | HitRate@5 | MRR@5 | Precision@5 | Joint loại+màu@5 | Candidate hit@100 | Thắng/thua/hòa vs B0 |
|---|---|---|---|---|---|---|
| A0_whole_image | 0.340 (16/47) | 0.214 (10.067/47) | 0.111 (26/235) | 0.431 (22/51) | 0.851 (40/47) | 7/16/24 |
| C1p_color_only | 0.213 (10/47) | 0.128 (6.033/47) | 0.064 (15/235) | 0.294 (15/51) | 0.383 (18/47) | 2/17/28 |
| C2_embedding_only | 0.362 (17/47) | 0.266 (12.500/47) | 0.187 (44/235) | 0.392 (20/51) | 0.511 (24/47) | 2/11/34 |
| B0_current | 0.426 (20/47) | 0.363 (17.083/47) | 0.221 (52/235) | 0.412 (21/51) | 0.511 (24/47) | 0/0/47 |
| B1_gamma | 0.426 (20/47) | 0.363 (17.083/47) | 0.217 (51/235) | 0.412 (21/51) | 0.511 (24/47) | 0/0/47 |
| C3_selected | 0.426 (20/47) | 0.355 (16.700/47) | 0.230 (54/235) | 0.412 (21/51) | 0.511 (24/47) | 1/4/42 |
| B2 (Mask thô) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B3 (Mask và morphology) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B4 (Cân bằng trắng và HSV) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| C1 (Histogram HSV) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| F (Cấu hình cuối) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

## Sau quyết định

| Cấu hình | HitRate@5 | MRR@5 | Coverage | False acceptance |
|---|---|---|---|---|
| A0_whole_image | 0.340 (16/47) | 0.214 (10.067/47) | 1.000 (51/51) | 4 ca / 4 |
| C1p_color_only | 0.213 (10/47) | 0.128 (6.033/47) | 0.510 (26/51) | 1 ca / 4 |
| C2_embedding_only | 0.149 (7/47) | 0.124 (5.833/47) | 0.176 (9/51) | 1 ca / 4 |
| B0_current | 0.000 (0/47) | 0.000 (0.000/47) | 0.000 (0/51) | 0 ca / 4 |
| B1_gamma | 0.000 (0/47) | 0.000 (0.000/47) | 0.000 (0/51) | 0 ca / 4 |
| C3_selected | 0.064 (3/47) | 0.064 (3.000/47) | 0.059 (3/51) | 0 ca / 4 |

## Phát hiện mục tiêu

| Cấu hình | Recall | Dark | Normal |
|---|---|---|---|
| B0_current | 0.519 (27/52) | 0.750 (3/4) | 0.500 (24/48) |
| B1_gamma | 0.519 (27/52) | 0.750 (3/4) | 0.500 (24/48) |
| A0_whole_image | n/a | n/a | n/a |
| C1p_color_only | 0.519 (27/52) | 0.750 (3/4) | 0.500 (24/48) |
| C2_embedding_only | 0.519 (27/52) | 0.750 (3/4) | 0.500 (24/48) |
| C3_selected | 0.519 (27/52) | 0.750 (3/4) | 0.500 (24/48) |

## Màu truy vấn

| Cấu hình | Độ chính xác |
|---|---|
| A0_whole_image | n/a |
| C1p_color_only | 0.222 (6/27) |
| C2_embedding_only | 0.222 (6/27) |
| B0_current | 0.222 (6/27) |
| B1_gamma | 0.222 (6/27) |
| C3_selected | 0.222 (6/27) |

| Cấu hình | Màu nhãn | Màu dự đoán | Số ca |
|---|---|---|---|
| C1p_color_only | black | black | 3 |
| C1p_color_only | black | brown | 1 |
| C1p_color_only | blue | cream | 1 |
| C1p_color_only | blue | gray | 1 |
| C1p_color_only | blue | navy | 1 |
| C1p_color_only | brown | black | 1 |
| C1p_color_only | brown | navy | 1 |
| C1p_color_only | gray | black | 1 |
| C1p_color_only | gray | cream | 1 |
| C1p_color_only | navy | black | 1 |
| C1p_color_only | navy | navy | 2 |
| C1p_color_only | olive | black | 1 |
| C1p_color_only | olive | brown | 2 |
| C1p_color_only | orange | brown | 1 |
| C1p_color_only | pink | cream | 1 |
| C1p_color_only | red | red | 1 |
| C1p_color_only | white | black | 1 |
| C1p_color_only | white | cream | 3 |
| C1p_color_only | white | gray | 3 |
| C2_embedding_only | black | black | 3 |
| C2_embedding_only | black | brown | 1 |
| C2_embedding_only | blue | cream | 1 |
| C2_embedding_only | blue | gray | 1 |
| C2_embedding_only | blue | navy | 1 |
| C2_embedding_only | brown | black | 1 |
| C2_embedding_only | brown | navy | 1 |
| C2_embedding_only | gray | black | 1 |
| C2_embedding_only | gray | cream | 1 |
| C2_embedding_only | navy | black | 1 |
| C2_embedding_only | navy | navy | 2 |
| C2_embedding_only | olive | black | 1 |
| C2_embedding_only | olive | brown | 2 |
| C2_embedding_only | orange | brown | 1 |
| C2_embedding_only | pink | cream | 1 |
| C2_embedding_only | red | red | 1 |
| C2_embedding_only | white | black | 1 |
| C2_embedding_only | white | cream | 3 |
| C2_embedding_only | white | gray | 3 |
| B0_current | black | black | 3 |
| B0_current | black | brown | 1 |
| B0_current | blue | cream | 1 |
| B0_current | blue | gray | 1 |
| B0_current | blue | navy | 1 |
| B0_current | brown | black | 1 |
| B0_current | brown | navy | 1 |
| B0_current | gray | black | 1 |
| B0_current | gray | cream | 1 |
| B0_current | navy | black | 1 |
| B0_current | navy | navy | 2 |
| B0_current | olive | black | 1 |
| B0_current | olive | brown | 2 |
| B0_current | orange | brown | 1 |
| B0_current | pink | cream | 1 |
| B0_current | red | red | 1 |
| B0_current | white | black | 1 |
| B0_current | white | cream | 3 |
| B0_current | white | gray | 3 |
| B1_gamma | black | black | 3 |
| B1_gamma | black | brown | 1 |
| B1_gamma | blue | cream | 1 |
| B1_gamma | blue | gray | 1 |
| B1_gamma | blue | navy | 1 |
| B1_gamma | brown | black | 1 |
| B1_gamma | brown | navy | 1 |
| B1_gamma | gray | black | 1 |
| B1_gamma | gray | cream | 1 |
| B1_gamma | navy | black | 1 |
| B1_gamma | navy | navy | 2 |
| B1_gamma | olive | black | 1 |
| B1_gamma | olive | brown | 2 |
| B1_gamma | orange | brown | 1 |
| B1_gamma | pink | cream | 1 |
| B1_gamma | red | red | 1 |
| B1_gamma | white | black | 1 |
| B1_gamma | white | cream | 3 |
| B1_gamma | white | gray | 3 |
| C3_selected | black | black | 3 |
| C3_selected | black | brown | 1 |
| C3_selected | blue | cream | 1 |
| C3_selected | blue | gray | 1 |
| C3_selected | blue | navy | 1 |
| C3_selected | brown | black | 1 |
| C3_selected | brown | navy | 1 |
| C3_selected | gray | black | 1 |
| C3_selected | gray | cream | 1 |
| C3_selected | navy | black | 1 |
| C3_selected | navy | navy | 2 |
| C3_selected | olive | black | 1 |
| C3_selected | olive | brown | 2 |
| C3_selected | orange | brown | 1 |
| C3_selected | pink | cream | 1 |
| C3_selected | red | red | 1 |
| C3_selected | white | black | 1 |
| C3_selected | white | cream | 3 |
| C3_selected | white | gray | 3 |

## Latency (ms)

| Cấu hình | Công đoạn | Ảnh đầu | Mean sau ảnh đầu | p95 sau ảnh đầu | n |
|---|---|---|---|---|---|
| A0_whole_image | preprocess | 88.038 | 19.520 | 27.097 | 51 |
| A0_whole_image | detect | 0.000 | 0.000 | 0.000 | 51 |
| A0_whole_image | embed | 113.308 | 59.742 | 74.557 | 51 |
| A0_whole_image | retrieve | 0.605 | 0.686 | 1.067 | 51 |
| A0_whole_image | rerank | 0.000 | 0.000 | 0.000 | 51 |
| A0_whole_image | total | 201.951 | 79.947 | 108.957 | 51 |
| C1p_color_only | preprocess | 88.038 | 19.520 | 27.097 | 51 |
| C1p_color_only | detect | 484.199 | 186.853 | 232.515 | 51 |
| C1p_color_only | embed | 115.727 | 111.510 | 290.604 | 51 |
| C1p_color_only | retrieve | 0.000 | 0.000 | 0.000 | 51 |
| C1p_color_only | rerank | 79.551 | 65.993 | 155.219 | 51 |
| C1p_color_only | total | 767.515 | 383.876 | 646.359 | 51 |
| C2_embedding_only | preprocess | 88.038 | 19.520 | 27.097 | 51 |
| C2_embedding_only | detect | 484.199 | 186.853 | 232.515 | 51 |
| C2_embedding_only | embed | 531.706 | 517.408 | 1320.306 | 51 |
| C2_embedding_only | retrieve | 4.922 | 4.845 | 13.164 | 51 |
| C2_embedding_only | rerank | 213.887 | 13.659 | 34.918 | 51 |
| C2_embedding_only | total | 1322.751 | 742.284 | 1596.124 | 51 |
| B0_current | preprocess | 88.038 | 19.520 | 27.097 | 51 |
| B0_current | detect | 484.199 | 186.853 | 232.515 | 51 |
| B0_current | embed | 531.706 | 517.408 | 1320.306 | 51 |
| B0_current | retrieve | 4.922 | 4.845 | 13.164 | 51 |
| B0_current | rerank | 3014.050 | 688.383 | 1820.224 | 51 |
| B0_current | total | 4122.914 | 1417.008 | 3316.871 | 51 |
| B1_gamma | preprocess | 84.372 | 22.366 | 32.908 | 51 |
| B1_gamma | detect | 174.438 | 182.774 | 212.581 | 51 |
| B1_gamma | embed | 531.265 | 513.371 | 1245.856 | 51 |
| B1_gamma | retrieve | 6.101 | 4.947 | 13.496 | 51 |
| B1_gamma | rerank | 758.633 | 410.615 | 1073.308 | 51 |
| B1_gamma | total | 1554.809 | 1134.074 | 2481.191 | 51 |
| C3_selected | preprocess | 88.038 | 19.520 | 27.097 | 51 |
| C3_selected | detect | 484.199 | 186.853 | 232.515 | 51 |
| C3_selected | embed | 531.706 | 517.408 | 1320.306 | 51 |
| C3_selected | retrieve | 4.922 | 4.845 | 13.164 | 51 |
| C3_selected | rerank | 343.982 | 377.217 | 950.509 | 51 |
| C3_selected | total | 1452.847 | 1105.842 | 2451.863 | 51 |

## Phân loại lỗi

| Cấu hình | no_detection | wrong_class | candidate_miss | rank_miss | rejected | hit |
|---|---|---|---|---|---|---|
| A0_whole_image | 0 | 0 | 7 | 24 | 0 | 16 |
| C1p_color_only | 8 | 14 | 7 | 8 | 0 | 10 |
| C2_embedding_only | 8 | 14 | 1 | 7 | 10 | 7 |
| B0_current | 8 | 14 | 1 | 4 | 20 | 0 |
| B1_gamma | 8 | 14 | 1 | 4 | 20 | 0 |
| C3_selected | 8 | 14 | 1 | 4 | 17 | 3 |
