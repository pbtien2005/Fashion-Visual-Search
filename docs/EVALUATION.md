# Kiểm thử và đánh giá

Chạy từ root trong môi trường đã cài dependencies. Không ghi đè evidence khóa ở `docs/m2/phase5/`; dùng output mới khi chạy lại.

## Suite offline

```powershell
python -B -m unittest discover -s tests -v
python -B -m evaluation.run_m2_checks --output output/m2/runs/checks_current
```

Lệnh thứ hai cần `coverage`, chạy cùng suite và lưu log/summary. Với test image local đã có đủ dependencies:

```powershell
docker run --rm --network none -v "${PWD}:/app" -w /app fashion-m2:phase3-tests python -B -m unittest discover -s tests -v
```

Đây là image local, không phải registry image phát hành. Fresh clone cần chuẩn bị dependencies module; `object_detection/Dockerfile.tests` chứa test dependencies detection, có thể cần bổ sung dependencies API/M3 để chạy toàn suite.

Suite thường dùng fixture encoder/DB, HTTP ASGI integration. Opt-in `RUN_REAL_HSV_INTEGRATION=1` cần pretrained FashionCLIP có sẵn; vẫn stage transport M2/DB, không phải live retrieval. Xem [M3_HSV_PRODUCTION](M3_HSV_PRODUCTION.md#tests-and-actual-limits).

## M2

```powershell
python -B -m evaluation.run_m2_eval --manifest evaluation/m2/annotations.review2.json --output output/m2/runs/review2_current --modes off raw open_close
```

Cần model `object_detection/best.pt`, thư viện detection, ảnh theo manifest. Review2 chứa 20 target dùng chung M1, 5 mask AI gợi ý/người dùng kiểm. Nhãn trên canvas working M1; không trộn upload/oriented/working. Checksum/provenance lưu trong manifest và `reference_masks/`.

Protocol/source fingerprints/cấu hình cuối ở `docs/m2/phase5/`. Runner `evaluation.run_m2_phase5` có stage `init`, `validation`, `test`; dùng `--output` mới để tái lập. `locked_config.json` cần khớp source trước test.

- Target recall 16/20 là tỷ lệ target được gán nhãn match theo evaluator, không phải mAP/recall mọi vật thể trong ảnh.
- Mask IoU chỉ tính reference hợp lệ; color similarity không phải retrieval sản phẩm.
- Tập nhỏ và đã xem ở phase 4, không gọi test là holdout độc lập. Lý do mask off/số liệu ở [M4_HANDOFF](M4_HANDOFF.md).

## Fixture hợp đồng

`tests/fixtures/m2/` giữ source RGB 96×128, detection khác pixel cùng geometry, tight ROI 64×64, mask chữ T tổng hợp, contact sheet và response JSON. `source_roi.png` là lát `[16:80,32:96]`, mask foreground=255. Candidate masked dùng reference synthetic; `phase3_responses.json` chứa output Otsu. `phase2_response.json` là serializer thực với bbox fixture, không phải YOLO ảnh thật. Các fixture legacy/no-detection giữ hành vi cũ.

```powershell
python -B -m evaluation.prepare_m2_phase1
python -B -m evaluation.check_m2_phase2 fixture
```

Chỉ sinh lại khi cố ý cập nhật fixture rồi review diff; không cần weights/DB/encoder/mạng. Synthetic không chứng minh chất lượng segmentation thật. Fixture M1 ở `tests/fixtures/preprocessing/` có EXIF và source/gamma alignment.

## Retrieval cho M4

Cần API chính, YOLO, encoder, DB có candidate đúng model/version, cache HSV production đã kiểm. API chạy thành công không đồng nghĩa tìm đúng sản phẩm.

```powershell
python evaluation/run_eval.py --api-url http://localhost:8000 --labels evaluation/labels.real.json --run-name real_outfits_current --timeout 300
python evaluation/run_benchmark.py --api-url http://localhost:8000 --run-prefix benchmark_current --timeout 300
python evaluation/compare_runs.py evaluation/results/previous.json evaluation/results/current.json
```

Label JSON là list query: `query_id`, `image` tương đối `evaluation/`, `expected_items`; mỗi item có `type`, `expected_categories`, tùy chọn `expected_color`/`acceptable_product_ids`. Dùng `labels.example.json`/`labels.real.example.json` làm mẫu, không ghi đè nhãn đã kiểm. Thiếu acceptable IDs thì Recall@5/MRR không được tính.

| Metric | Cách hiểu |
|---|---|
| `detection_success_rate` | Ảnh có ít nhất một crop, không phải detector recall/mAP |
| `category_at_5`, `color_at_5` | Metadata kỳ vọng xuất hiện trong top 5, matching đơn giản |
| `recall_at_5`, `mrr` | Product ID chấp nhận được trong top 5 / reciprocal rank đầu tiên |
| `avg_latency_ms`, `failed_queries` | Tốc độ/lỗi request của lần chạy |

`auto_smoke` dùng ảnh catalog; `starter_catalog` regression với product ID; `real_outfits` gần ảnh người dùng hơn. Aggregate suite có trọng số .20/.30/.50; score/gates là tiêu chí kỹ thuật harness, không phải accuracy chung. Xem `run_benchmark.py` cho công thức/gates. Không lấy benchmark trước HSV làm số đo hiện tại.

`tune_rerank_weights.py` chỉ rerank candidate đã trả, không đo sản phẩm bị mất từ truy hồi. Tune validation rồi chạy API trên test phù hợp. Tên màu có giới hạn của nhãn HSV xấp xỉ; relevance cần kiểm độc lập.

Builder `evaluation/build_wikimedia_real_eval.py` tải ảnh và ghi lại dataset/nhãn; chỉ chạy khi chủ động tái tạo dataset, không phải kiểm thử offline thông thường. Xem [DATA_SOURCES](DATA_SOURCES.md).
