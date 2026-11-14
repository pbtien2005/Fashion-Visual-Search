# Review merge M3 và điều chỉnh kế hoạch M2 — 09/11/2026

Review tại merge `2962f21`, M2 trước merge `500bedf`; M3 gồm `5a433e9` và `4562a8f`. Working tree sạch trước lượt review. `object_detection/` không thay đổi qua merge, nên giữ nghiệm thu phase 1–2 và weights; không làm lại YOLO/ROI.

## Kết luận

Giữ số phase 1–6. **Thêm bước nối hợp đồng ở đầu phase 3**, sau đó Otsu/morphology như cũ. Phase 4 tái sử dụng công cụ màu M3; phase 5 rút gọn phần xây/nối M3 đã có nhưng phải rebase đối chứng trên HSV147 đã merge. Nội dung thực hiện khi gõ “làm phase n” đã được cập nhật trong `M2_PLAN.md`.

Lượt này chỉ thay kế hoạch/contract, thêm công cụ và bằng chứng audit; chưa sửa code production M3 hay triển khai phase 3.

## Những gì tái sử dụng được

- `AI/hsv_descriptor.py`: histogram H-S 18×8 + Black/Gray/White, 147 chiều, version/config, mask validation và Hellinger similarity.
- `AI/features.py`: FeatureBundle, encoder và ROI màu tách riêng; các import encoder được lazy ở điểm cần.
- `AI/production_color.py`: dùng chung query/catalog, missing-color policy, cache và swatch hiển thị.
- `api/main.py`, `api/search_service.py`, `AI/vectorizer.py`: đã nối HSV vào runtime/query/catalog. Một descriptor màu được dùng lại cho các JPEG variants.
- `AI/color_cache.py`, `AI/rebuild_color_cache.py`, `evaluation/run_color_benchmark.py`: cache màu riêng và công cụ thực nghiệm. Không cần viết lại những phần này.

## Phát hiện và tác động

| Ưu tiên | Phát hiện ở code merge | Tác động/kế hoạch |
|---|---|---|
| Cao | `color_input_from_detection` chỉ đọc `source_roi_base64`/`mask_base64`; M2 gửi `source_crop_png_base64`/`mask_png_base64` | ROI nguồn và mask M2 không được tiêu thụ. Sửa canonical key priority, giữ alias cũ ở phase 3 |
| Cao | Fallback M3 ưu tiên `original_bbox` int cũ thay vì ROI PNG/bbox floor-ceil của M2 | Có thể khác ROI 1 pixel mỗi chiều, mask sai shape rồi fallback âm thầm. Dùng PNG làm nguồn chuẩn; không resize để che sai lệch |
| Vừa | Runtime tạo feature ID từ `crop.id` hoặc class name, không ưu tiên `garment_id` | Nhiều áo có thể chung identity trong provenance. Dùng ID M2 ở glue; evaluator vẫn ghép đúng target bằng class+bbox |
| Cao cho thí nghiệm | Runtime đã đổi RGB sang HSV; color penalty cũ thành 0; display labels HSV có thể đi vào text/style | Không dùng baseline trước merge để quy chênh lệch cho mask. Tạo H0 sau merge/adapter fix, rồi B2/B3 chỉ thay mask |
| Cao cho retrieval | Missing-color mặc định `zero`; default sidecar chưa có trên workspace | Nếu thiếu cache, color score 0 có thể làm che tác dụng mask. Kiểm coverage/version, dùng cache cố định hoặc chỉ đo màu/IoU offline |
| Vừa | M3 đọc mask_status nhưng chưa nhận nguyên metadata M2 | Giữ source/bbox/geometry/config/fallback M2 trong metadata để audit mask có được dùng thật |

Không đổi producer M2 sang tên key cũ chỉ để khớp decoder. Ưu tiên hợp đồng phase 2 đã kiểm; alias chỉ để tương thích. Không thay HSVConfig trong thí nghiệm M2. Sau M1, bbox working không được cắt trên ảnh upload khác hình học; canonical ROI PNG giải quyết đúng ranh giới này.

## Audit chạy thật, không cần model/DB

`evaluation/check_m2_m3_contract.py` chạy trực tiếp decoder và HSV extractor M3 trên fixture M2 hiện có. Kết quả tại `docs/m2/m3_merge_review_20261109/evidence.json`:

- M2 PNG đúng 64×64 bị bỏ qua; không truyền original thì nhận crop legacy 224×224.
- Với original ảnh và legacy bbox làm tròn khác, nhận 63×63 thay vì PNG 64×64.
- Mask PNG canonical bị bỏ qua: `mask_status=missing`, lấy 4.096 pixel thay vì 2.816 pixel foreground của fixture.
- Đổi tên cùng payload sang alias M3 thì ROI/mask và descriptor đều đúng. Điều này chỉ ra lỗi nối contract; không kết luận extractor HSV sai.

Audit trả trạng thái **needs_alignment** (nonzero có chủ đích). Gate phase 3 là chuyển sang **compatible**, gồm cả giữ alias hoạt động. Không chỉnh fixture sang alias để làm gate giả qua.

Đã chạy lại **53 test liên quan, đều qua**: 27 detection/model/API M2 + 6 fixture contract + 17 histogram M3 + 3 adapter M3 theo giao diện cũ. Điều này giải thích vì sao test từng module có thể qua nhưng joint contract vẫn lỗi. Chưa chạy toàn suite API/encoder hoặc FashionCLIP/DB/live retrieval; báo cáo 72 test của M3 là bằng chứng lịch sử trước merge, không phải kết quả lượt này.

Lệnh audit từ PowerShell root, dùng image CPU đã có:

```powershell
$m2Bind = "type=bind,source=$((Get-Location).Path),target=/workspace"
docker run --rm --network none --mount $m2Bind -w /workspace --entrypoint python fashion-m2:phase1 -m evaluation.check_m2_m3_contract --revision 2962f21
```

Lệnh chỉ chạy fixture/HSV, không tải pretrained model hoặc cập nhật cache/DB. Khi sửa ở phase 3, lưu kết quả mới riêng; giữ evidence review này làm mốc trước sửa.

## Cache và số liệu

Tại review không có `data/color_cache_hsv147.json`; chưa đọc override riêng trong `.env`, nên chưa kết luận deployment đang thiếu cache. Bằng chứng M3 có sample 28 ảnh `--skip-detection`/whole-image fallback và cache mẫu lịch sử; đó không phải coverage full catalog với garment mask.

H0/B2/B3 đầu tiên giữ cùng catalog cache và chỉ thay query mask. Nếu áp dụng mask catalog, rebuild màu ra cache riêng theo cấu hình, ghi nguồn ROI/mask và kiểm coverage; không re-embedding nếu nhánh JPEG/encoder không đổi. Nếu mask làm đổi nhãn màu catalog, giữ text/style scores/labels cố định khi cần tách hiệu ứng xếp hạng. Không điều chỉnh trọng số cho đẹp số liệu M2.

M3 có thể đo màu sau nối adapter mà không DB; dùng công cụ đó để giảm công việc. M2 vẫn cần nhãn bbox/mask người kiểm tra, target matching và IoU; không lấy nhãn màu/ảnh tổng hợp của M3 thay cho chúng.

## Ưu tiên đến hạn nộp

- 09/11: phase 3 nối contract trước, rồi Otsu/morphology; bắt đầu chuẩn bị nhãn phase 4.
- 10/11: hoàn tất metric/đối chứng trên cùng dữ liệu, viết phần báo cáo và slide, kiểm tra bàn giao.
- Giữ nguyên yêu cầu 20 ảnh phần Người 2 và 5 mask theo kế hoạch; thiếu phải ghi rõ, không tự điền pseudo-label thành GT.
- Không thêm model, fine-tune, descriptor hay migration. Không nhận làm lại toàn bộ M3/M4.
