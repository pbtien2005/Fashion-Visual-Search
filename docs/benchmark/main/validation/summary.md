# Benchmark validation

## Tập dữ liệu

- 23 ảnh. has_match 22, no_match 1, unconfirmed 0.
- Lớp · long_sleeve_top 1, short_sleeve_dress 3, short_sleeve_top 8, shorts 3, skirt 3, trousers 4, vest_dress 1.
- Ánh sáng · dark 6, normal 17.
- Màu · black 9, blue 2, brown 2, gray 1, green 3, navy 1, olive 1, pink 1, red 1, white 2.

## Trước ngưỡng

C1p_color_only là đối chứng màu trội và từ khóa. C1 của nhóm dùng histogram HSV.
HitRate chỉ đo có sản phẩm chấp nhận được trong top-k. Recall cần biết toàn bộ sản phẩm phù hợp. Candidate hit@100 là cận dưới trong phạm vi đã chấm.

| Cấu hình | HitRate@5 | MRR@5 | Precision@5 | Joint loại+màu@5 | Candidate hit@100 | Thắng/thua/hòa vs B0 |
|---|---|---|---|---|---|---|
| A0_whole_image | 0.227 (5/22) | 0.114 (2.500/22) | 0.064 (7/110) | 0.304 (7/23) | 0.818 (18/22) | 4/10/8 |
| C1p_color_only | 0.227 (5/22) | 0.121 (2.667/22) | 0.073 (8/110) | 0.478 (11/23) | 0.727 (16/22) | 3/8/11 |
| C2_embedding_only | 0.455 (10/22) | 0.311 (6.833/22) | 0.209 (23/110) | 0.435 (10/23) | 0.727 (16/22) | 3/3/16 |
| B0_current | 0.500 (11/22) | 0.305 (6.700/22) | 0.218 (24/110) | 0.478 (11/23) | 0.727 (16/22) | 0/0/22 |
| B1_gamma | 0.500 (11/22) | 0.305 (6.700/22) | 0.218 (24/110) | 0.478 (11/23) | 0.727 (16/22) | 0/0/22 |
| C3_w1 | 0.500 (11/22) | 0.339 (7.450/22) | 0.218 (24/110) | 0.478 (11/23) | 0.727 (16/22) | 1/1/20 |
| C3_w2 | 0.455 (10/22) | 0.295 (6.500/22) | 0.218 (24/110) | 0.435 (10/23) | 0.727 (16/22) | 0/1/21 |
| B2 (Mask thô) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B3 (Mask và morphology) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B4 (Cân bằng trắng và HSV) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| C1 (Histogram HSV) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| F (Cấu hình cuối) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

## Sau quyết định

| Cấu hình | HitRate@5 | MRR@5 | Coverage | False acceptance |
|---|---|---|---|---|
| A0_whole_image | 0.227 (5/22) | 0.114 (2.500/22) | 1.000 (23/23) | 1 ca / 1 |
| C1p_color_only | 0.227 (5/22) | 0.121 (2.667/22) | 0.739 (17/23) | 1 ca / 1 |
| C2_embedding_only | 0.227 (5/22) | 0.140 (3.083/22) | 0.391 (9/23) | 1 ca / 1 |
| B0_current | 0.000 (0/22) | 0.000 (0.000/22) | 0.000 (0/23) | 0 ca / 1 |
| B1_gamma | 0.000 (0/22) | 0.000 (0.000/22) | 0.000 (0/23) | 0 ca / 1 |
| C3_w1 | 0.045 (1/22) | 0.011 (0.250/22) | 0.043 (1/23) | 0 ca / 1 |
| C3_w2 | 0.000 (0/22) | 0.000 (0.000/22) | 0.000 (0/23) | 0 ca / 1 |

## Phát hiện mục tiêu

| Cấu hình | Recall | Dark | Normal |
|---|---|---|---|
| B0_current | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |
| B1_gamma | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |
| A0_whole_image | n/a | n/a | n/a |
| C1p_color_only | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |
| C2_embedding_only | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |
| C3_w1 | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |
| C3_w2 | 0.739 (17/23) | 0.667 (4/6) | 0.765 (13/17) |

## Màu truy vấn

| Cấu hình | Độ chính xác |
|---|---|
| A0_whole_image | n/a |
| C1p_color_only | 0.353 (6/17) |
| C2_embedding_only | 0.353 (6/17) |
| B0_current | 0.353 (6/17) |
| B1_gamma | 0.353 (6/17) |
| C3_w1 | 0.353 (6/17) |
| C3_w2 | 0.353 (6/17) |

| Cấu hình | Màu nhãn | Màu dự đoán | Số ca |
|---|---|---|---|
| C1p_color_only | black | black | 4 |
| C1p_color_only | black | navy | 3 |
| C1p_color_only | blue | navy | 1 |
| C1p_color_only | brown | tan | 1 |
| C1p_color_only | gray | navy | 1 |
| C1p_color_only | green | navy | 1 |
| C1p_color_only | green | olive | 1 |
| C1p_color_only | green | yellow | 1 |
| C1p_color_only | olive | olive | 1 |
| C1p_color_only | red | red | 1 |
| C1p_color_only | white | cream | 1 |
| C1p_color_only | white | olive | 1 |
| C2_embedding_only | black | black | 4 |
| C2_embedding_only | black | navy | 3 |
| C2_embedding_only | blue | navy | 1 |
| C2_embedding_only | brown | tan | 1 |
| C2_embedding_only | gray | navy | 1 |
| C2_embedding_only | green | navy | 1 |
| C2_embedding_only | green | olive | 1 |
| C2_embedding_only | green | yellow | 1 |
| C2_embedding_only | olive | olive | 1 |
| C2_embedding_only | red | red | 1 |
| C2_embedding_only | white | cream | 1 |
| C2_embedding_only | white | olive | 1 |
| B0_current | black | black | 4 |
| B0_current | black | navy | 3 |
| B0_current | blue | navy | 1 |
| B0_current | brown | tan | 1 |
| B0_current | gray | navy | 1 |
| B0_current | green | navy | 1 |
| B0_current | green | olive | 1 |
| B0_current | green | yellow | 1 |
| B0_current | olive | olive | 1 |
| B0_current | red | red | 1 |
| B0_current | white | cream | 1 |
| B0_current | white | olive | 1 |
| B1_gamma | black | black | 4 |
| B1_gamma | black | navy | 3 |
| B1_gamma | blue | navy | 1 |
| B1_gamma | brown | tan | 1 |
| B1_gamma | gray | navy | 1 |
| B1_gamma | green | navy | 1 |
| B1_gamma | green | olive | 1 |
| B1_gamma | green | yellow | 1 |
| B1_gamma | olive | olive | 1 |
| B1_gamma | red | red | 1 |
| B1_gamma | white | cream | 1 |
| B1_gamma | white | olive | 1 |
| C3_w1 | black | black | 4 |
| C3_w1 | black | navy | 3 |
| C3_w1 | blue | navy | 1 |
| C3_w1 | brown | tan | 1 |
| C3_w1 | gray | navy | 1 |
| C3_w1 | green | navy | 1 |
| C3_w1 | green | olive | 1 |
| C3_w1 | green | yellow | 1 |
| C3_w1 | olive | olive | 1 |
| C3_w1 | red | red | 1 |
| C3_w1 | white | cream | 1 |
| C3_w1 | white | olive | 1 |
| C3_w2 | black | black | 4 |
| C3_w2 | black | navy | 3 |
| C3_w2 | blue | navy | 1 |
| C3_w2 | brown | tan | 1 |
| C3_w2 | gray | navy | 1 |
| C3_w2 | green | navy | 1 |
| C3_w2 | green | olive | 1 |
| C3_w2 | green | yellow | 1 |
| C3_w2 | olive | olive | 1 |
| C3_w2 | red | red | 1 |
| C3_w2 | white | cream | 1 |
| C3_w2 | white | olive | 1 |

## Trọng số đã thử

| Cấu hình | Ảnh | Loại | Màu | Text/style | MRR@5 |
|---|---|---|---|---|---|
| B0_current | 0.600 | 0.150 | 0.150 | 0.100 | 0.305 (6.700/22) |
| C3_w1 | 0.700 | 0.100 | 0.100 | 0.100 | 0.339 (7.450/22) |
| C3_w2 | 0.500 | 0.200 | 0.200 | 0.100 | 0.295 (6.500/22) |

C3_selected · C3_w1

Cấu hình đã khóa được giữ theo lượt test.

## Latency (ms)

| Cấu hình | Công đoạn | Ảnh đầu | Mean sau ảnh đầu | p95 sau ảnh đầu | n |
|---|---|---|---|---|---|
| A0_whole_image | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| A0_whole_image | detect | 0.000 | 0.000 | 0.000 | 22 |
| A0_whole_image | embed | 118.559 | 57.294 | 64.048 | 22 |
| A0_whole_image | retrieve | 0.623 | 0.714 | 1.594 | 22 |
| A0_whole_image | rerank | 0.000 | 0.000 | 0.000 | 22 |
| A0_whole_image | total | 168.168 | 83.464 | 113.260 | 22 |
| C1p_color_only | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| C1p_color_only | detect | 527.168 | 184.278 | 233.688 | 22 |
| C1p_color_only | embed | 77.381 | 151.609 | 433.077 | 22 |
| C1p_color_only | retrieve | 0.000 | 0.000 | 0.000 | 22 |
| C1p_color_only | rerank | 68.628 | 85.600 | 206.003 | 22 |
| C1p_color_only | total | 722.163 | 446.942 | 845.452 | 22 |
| C2_embedding_only | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| C2_embedding_only | detect | 527.168 | 184.278 | 233.688 | 22 |
| C2_embedding_only | embed | 381.065 | 713.048 | 2059.676 | 22 |
| C2_embedding_only | retrieve | 3.447 | 7.009 | 20.230 | 22 |
| C2_embedding_only | rerank | 209.668 | 20.034 | 51.261 | 22 |
| C2_embedding_only | total | 1170.334 | 949.825 | 2338.081 | 22 |
| B0_current | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| B0_current | detect | 527.168 | 184.278 | 233.688 | 22 |
| B0_current | embed | 381.065 | 713.048 | 2059.676 | 22 |
| B0_current | retrieve | 3.447 | 7.009 | 20.230 | 22 |
| B0_current | rerank | 2365.848 | 1103.086 | 1899.348 | 22 |
| B0_current | total | 3326.514 | 2032.877 | 4215.864 | 22 |
| B1_gamma | preprocess | 56.167 | 32.344 | 62.912 | 22 |
| B1_gamma | detect | 189.714 | 174.114 | 231.835 | 22 |
| B1_gamma | embed | 569.136 | 681.364 | 1934.055 | 22 |
| B1_gamma | retrieve | 6.866 | 6.594 | 18.904 | 22 |
| B1_gamma | rerank | 1009.299 | 547.504 | 1538.625 | 22 |
| B1_gamma | total | 1831.181 | 1441.920 | 3645.415 | 22 |
| C3_w1 | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| C3_w1 | detect | 527.168 | 184.278 | 233.688 | 22 |
| C3_w1 | embed | 381.065 | 713.048 | 2059.676 | 22 |
| C3_w1 | retrieve | 3.447 | 7.009 | 20.230 | 22 |
| C3_w1 | rerank | 271.773 | 501.240 | 1410.707 | 22 |
| C3_w1 | total | 1232.439 | 1431.030 | 3697.527 | 22 |
| C3_w2 | preprocess | 48.986 | 25.455 | 50.173 | 22 |
| C3_w2 | detect | 527.168 | 184.278 | 233.688 | 22 |
| C3_w2 | embed | 381.065 | 713.048 | 2059.676 | 22 |
| C3_w2 | retrieve | 3.447 | 7.009 | 20.230 | 22 |
| C3_w2 | rerank | 254.458 | 514.780 | 1593.279 | 22 |
| C3_w2 | total | 1215.123 | 1444.571 | 3880.099 | 22 |

## Phân loại lỗi

| Cấu hình | no_detection | wrong_class | candidate_miss | rank_miss | rejected | hit |
|---|---|---|---|---|---|---|
| A0_whole_image | 0 | 0 | 4 | 13 | 0 | 5 |
| C1p_color_only | 3 | 3 | 0 | 11 | 0 | 5 |
| C2_embedding_only | 3 | 3 | 0 | 6 | 5 | 5 |
| B0_current | 3 | 3 | 0 | 5 | 11 | 0 |
| B1_gamma | 3 | 3 | 0 | 5 | 11 | 0 |
| C3_w1 | 3 | 3 | 0 | 5 | 10 | 1 |
| C3_w2 | 3 | 3 | 0 | 6 | 10 | 0 |
