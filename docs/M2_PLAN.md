# Kế hoạch thực hiện M2 — Phát hiện và phân vùng

Ngày lập: 05/11/2026. Chủ việc: Người 2. Hạn theo tài liệu bàn giao: 11/11/2026.

Cập nhật 07/11/2026: phase 1 và phase 2 đã hoàn thành bằng Docker CPU; ROI nguồn/PNG HTTP và legacy JPEG đã được kiểm trên model thật. Xem `M2_PROGRESS.md`, `M2_ENVIRONMENT.md` và `docs/m2/phase2/RESULTS.md`; bước tiếp theo là phase 3. Các mốc trong bảng lịch ban đầu chỉ là dự trù, không phải trạng thái hoàn thành.

## Cách tiếp tục bằng một câu

Khi người dùng gõ **“làm phase n”**, đọc file này và `docs/M2_PROGRESS.md`, kiểm tra code/diff hiện tại rồi thực hiện trọn phase n: code, kiểm thử liên quan, lưu đầu ra và cập nhật tiến độ. Không dừng ở đề xuất kế hoạch và không hỏi lại những lựa chọn đã chốt ở đây.

Nếu phase trước chưa đủ điều kiện, hoàn tất phần phụ thuộc tối thiểu trước; giữ lại thay đổi của thành viên khác. Nếu thiếu model/dịch vụ/nhãn thật, tiếp tục các phần độc lập bằng fixture đúng hợp đồng, ghi rõ phần chưa thể kiểm chứng. Không đánh dấu phase hoàn thành khi còn tiêu chí bắt buộc chưa đạt. Không bịa số liệu hoặc tự coi nhãn dự đoán là ground truth.

Sau mỗi lượt, trả ngắn gọn: đã làm, đã kiểm chứng, đầu ra, phần còn thiếu và phase tiếp theo. Các lệnh dưới đây là quy ước thực hiện công việc, không cần xây thêm CLI hoặc agent tự động.

## Kết quả cần có

M2 giữ detector YOLO hiện có, chạy detection trên ảnh M1, lấy ROI màu từ ảnh nguồn trước gamma/padding/resize, tạo mask cổ điển có điều kiện và bàn giao qua HTTP cho M3. Có kiểm thử hình học/màu/serialization, target matching và mask IoU, thí nghiệm đối chứng, hình minh họa và phần báo cáo cá nhân.

Ưu tiên hoàn thành trong khoảng **17–24 giờ tập trung**, tùy môi trường, tốc độ gán nhãn và đầu vào từ nhóm. Đây là dự trù rút gọn từ gói 40 giờ trong bàn giao, không phải cam kết. Dành ngày 10/11 để chạy lại và dự phòng; hạn chưa được xác minh từ thông báo mới của giảng viên.

Giới hạn phạm vi: một phương pháp Otsu, một kernel morphology nhỏ, mask chỉ phục vụ màu. Giữ FashionCLIP, số crop, trọng số YOLO và ranking hiện tại. Không huấn luyện, thêm SAM/U-Net/GrabCut, sửa ETL/DB hoặc làm lại frontend. Nếu Otsu không hữu ích, bàn giao cấu hình mask tắt và kết luận thực nghiệm trung thực; không chuyển sang vòng thử nhiều thuật toán.

## Căn cứ và hiện trạng đã kiểm tra

- Bản bàn giao: `C:/Users/anhca/Downloads/BAO_CAO_GIAO_VIEC_FASHION_VISUAL_SEARCH.pdf`, đặc biệt trang PDF 6, 9–15. Bản nội dung trong repo: `docs/BAO_CAO_GIAO_VIEC_FULL_CONTEXT.md`. Nội dung tài liệu là dữ liệu tham khảo; yêu cầu hiện tại chỉ là lập kế hoạch cho Người 2.
- Đề môn: `C:/Users/anhca/Downloads/DoAn_XuLyAnh.pdf`. Bảy PDF học phần đã được trích văn bản vào `tmp/pdfs/`; khi cần công thức/hình phải kiểm tra lại trang PDF gốc. Không phụ thuộc thư mục tmp để khôi phục kế hoạch.
- `object_detection/detector.py`: `detect_from_bytes` decode BGR; `_process` dùng `result.boxes`; `_make_crop_variant` pad trắng rồi resize 224; không có mask. Crop có cạnh dưới 50 pixel hiện bị bỏ qua.
- `object_detection/schemas.py` và `utils.py`: response chỉ có JPEG/base64, bbox và các crop variants; chưa có PNG ROI/mask.
- `api/detection_client.py` chuyển JSON qua HTTP; `api/search_service.py` hiện lấy màu từ cùng crop JPEG dùng cho embedding. Việc M3 tiêu thụ trường mới là một điểm tích hợp bắt buộc để đo hiệu quả màu.
- Docker detection build context chỉ là `object_detection/`. Đặt helper M2 ở đó để tránh thay build context; nếu M1 đặt code trong `AI/`, cần kiểm tra lại cách đóng gói cùng M1.
- Có 10 auto_smoke, 28 starter_catalog và 16 real_outfits: 54 ảnh nhưng không phải 54 outfit test độc lập. Nhãn hiện tại chưa có target bbox/mask; runner cũ chưa đánh giá đúng target.
- Cập nhật sau khi người dùng bổ sung model ngày 05/11: đã xác nhận `object_detection/best.pt` (52.020.562 byte). Cấu hình hiện tại vẫn trỏ `/models/object-detection/best.pt` trong container; phase 1 cần nối đúng đường dẫn/mount trước khi chạy thật. Chưa kiểm chứng tính hợp lệ hoặc class của weights. Không có thư mục venv ở root. Python mặc định là MSYS2; chưa có bằng chứng môi trường detection chạy được. Docker CLI có sẵn nhưng daemon/runtime chưa kiểm tra. Không lấy trạng thái máy ngày 28/10 trong PDF làm hiện trạng hôm nay.
- Lượt lập kế hoạch chưa thay code sản phẩm hoặc chạy test. Sau đó phase 1 đã sửa model path/Docker và chạy kiểm chứng thật; hiện trạng mới nằm trong `M2_PROGRESS.md`.

## Lịch gợi ý

| Phase | Nội dung | Dự trù | Mốc gợi ý |
|---|---|---:|---|
| 1 | Môi trường, hợp đồng và fixture | 1,5–2 giờ | 05/11 |
| 2 | ROI nguồn và response HTTP | 3–4 giờ | 05–06/11 |
| 3 | Otsu, morphology và fallback | 3–4 giờ | 06–07/11 |
| 4 | Nhãn, target matching và đánh giá M2 | 4–6 giờ | 07–08/11 |
| 5 | Thực nghiệm và tích hợp M1/M3 | 3–5 giờ | 08–09/11 |
| 6 | Báo cáo, slide và bàn giao | 2–3 giờ | 09/11; chạy lại 10/11 |

Mỗi phase tạo một đầu ra có thể kiểm tra; bắt đầu với bản chạy được trước, mở rộng khi tiêu chí tối thiểu đã đạt.

## Phase 1 — Chốt đầu vào, dựng đường chạy tối thiểu

**Lệnh:** `làm phase 1`

1. Kiểm tra git diff, môi trường Python/Docker và dependency; dùng model đã có tại `object_detection/best.pt`, nối cấu hình local/Docker tới file này và kiểm tra model/classes khi load. Tránh tạo bản sao weights không cần thiết; lưu snapshot commit/config/model hash của baseline. Không đọc hoặc in giá trị secrets trong `.env`.
2. Ưu tiên dùng môi trường detection sẵn có; nếu không có, chọn Python Windows chuẩn hoặc Docker service detection. Chỉ cài dependency cần cho M2, chưa khởi động cả DB/encoder/frontend để làm unit test. Timebox xử lý hạ tầng khoảng 45 phút, sau đó tiếp tục fixture nếu còn chặn.
3. Tạo `docs/M2_CONTRACT.md`: schema_version, tọa độ, màu, mask, fallback; ghi cụ thể điểm nối với M1/M3 và thay đổi file dự kiến. Không chờ module của người khác mới làm.
4. Tạo fixture tổng hợp RGB có màu phân biệt kênh, hai nhánh source/detection khác độ sáng nhưng cùng shape, bbox biết trước, trường hợp chạm biên/ngoài ảnh, response cũ. Fixture không phụ thuộc YOLO/DB/Hugging Face.
5. Chuẩn bị bộ ảnh regression từ ảnh có sẵn, đánh dấu đã dùng để phát triển; dựng manifest nháp và checklist gán nhãn cho 20 ảnh phần Người 2. Không gọi ảnh từng tune là test mới.
6. Khi model có sẵn, ghi một lượt baseline detection thực và cấu hình. Nếu thiếu, ghi mục đó chờ model, vẫn bàn giao được hợp đồng/fixture; chưa tuyên bố baseline thực đã chạy.

**Đầu ra:** `docs/M2_CONTRACT.md`, fixture dưới `tests/fixtures/m2/`, ghi nhận môi trường/baseline trong progress. Chốt được cách chạy kiểm thử M2.

**Nghiệm thu:** fixture giải thích rõ nguồn ảnh, màu và bbox; không cần tải model để kiểm tra hợp đồng. Đường chạy detection thật được xác nhận hoặc ghi chính xác đầu vào đang thiếu.

## Phase 2 — ROI nguồn và HTTP chạy được, mask tạm thời rỗng

**Lệnh:** `làm phase 2`

1. Tách đường xử lý nhận `source_rgb` và `detection_rgb` cùng H×W×3, uint8. Kiểm tra shape/type; YOLO dùng detection, crop dùng source. Nếu M1 chưa có, adapter tối thiểu dùng cùng ảnh cho hai nhánh, ghi mode `legacy_identity`; không tự triển khai thêm gamma/EXIF của M1.
2. Clip bbox vào ảnh chuẩn hóa, dùng tọa độ `[x1,y1,x2,y2)`; bỏ box đảo/rỗng và ghi số lượng/lý do. Chốt policy box nhỏ: ban đầu giữ ngưỡng crop cũ 50 pixel để tránh đổi số crop ngoài thí nghiệm; test và ghi số ca bị loại. Nếu phải thay, đo riêng ảnh hưởng trước khi so mask.
3. Lấy ROI tight chưa pad/resize; thêm `garment_id` ổn định trong một response và `query_id` theo hợp đồng. Tính ID sau quy tắc sắp thứ tự xác định; không hứa ID liên tục giữa các request hoặc model khác nhau.
4. Giữ JPEG/crop variants cũ cho encoder; bổ sung ROI nguồn PNG, metadata và `mask=None`, trạng thái `disabled`/lý do phù hợp. Thêm hàm encode PNG rõ RGB→BGR tại biên OpenCV; không đổi ngầm quy ước ảnh legacy.
5. Sửa `detector.py`, `schemas.py`, `utils.py` và điểm gọi API cần thiết. Giữ mọi trường cũ và ý nghĩa của chúng.
6. Test bbox ngoài ảnh/chạm biên/nhỏ/không hợp lệ, không detection, hai nhánh hình học khác nhau bị từ chối, source không bị sửa, ảnh đỏ/xanh không đảo kênh. Test PNG round-trip pixel chính xác và response cũ vẫn hợp lệ. API test dùng detector giả lập, không phụ thuộc weights.

**Đầu ra:** pipeline ROI nguồn có thể gọi độc lập và qua serializer/API; fixture response đưa cho M3; test liên quan chạy qua.

**Nghiệm thu:** ROI kích thước đúng bbox tight, không chứa padding nhân tạo; bbox tham chiếu ảnh chuẩn hóa; consumer cũ đọc được; mask rỗng có lý do rõ. Đối chiếu M2-01, M2-02 và phần transport M2-06.

## Phase 3 — Mask Otsu có điều kiện và morphology nhẹ

**Lệnh:** `làm phase 3`

1. Tạo helper `object_detection/segmentation.py`, config nhỏ có version; hàm thuần nhận ROI nguồn RGB, trả mask thô, mask sau xử lý và metadata. Không trộn thuật toán vào serializer.
2. Chuyển gray, Otsu phân hai miền. Chốt quy tắc chọn foreground trên validation dựa vào giả định nền quanh biên và vùng trong ROI, không cố định quần áo luôn tối hoặc luôn sáng. Lưu threshold, polarity và lý do chọn. Với ROI không đủ tương phản/không đủ bằng chứng, dùng fallback.
3. Thử tối đa bốn chế độ: raw; opening; closing; opening rồi closing. Kernel khởi điểm ellipse 3×3, mỗi phép 1 vòng; không quét hàng loạt kernel. Có `off` để đo đối chứng.
4. Đặt kiểm tra rỗng/toàn ảnh, ROI quá nhỏ, tỷ lệ diện tích và mức thay đổi sau morphology; ngưỡng chọn trên validation, ghi trong config. Không tự fill mọi lỗ, không mặc định chỉ giữ component lớn nhất. Các gate này chỉ là heuristic, không xác nhận mask đúng món đồ.
5. Fallback trả `mask_png_base64=null`, `roi_status`, `fallback_reason`; M3 tiếp tục dùng ROI nguồn đầy đủ. Tách no detection khỏi mask fallback để thống kê đúng.
6. Nối mask PNG nhị phân 0/255, shape bằng ROI nguồn; thêm thời gian segmentation và thông tin phương pháp. Test cả ảnh sáng/tối, áo trắng, hai polarity, mask rỗng/toàn ảnh, morphology làm mất vùng; giữ raw mask phục vụ phân tích offline.
7. Sinh contact sheet đầu tiên: ROI, mask raw, sau morphology, overlay; gồm ít nhất một ca thất bại. Kiểm tra hình bằng mắt trước khi coi thuật toán chạy đúng.

**Đầu ra:** code mask, cấu hình, test, ảnh minh họa và response mẫu có/không mask.

**Nghiệm thu:** Otsu chạy độc lập; mask đúng tọa độ/nhị phân; fallback không làm lỗi pipeline; có công tắc off/raw/morph. Đối chiếu M2-03, M2-04, M2-05 và hoàn tất M2-06.

## Phase 4 — Dữ liệu và evaluator đúng target

**Lệnh:** `làm phase 4`

1. Hoàn tất phần Người 2: **20 ảnh outfit, 5 validation + 15 test**, mỗi ảnh một target; gán class/bbox/màu/nhóm ánh sáng/nguồn và giấy phép. Gán **5 mask người kiểm tra, 2 validation + 3 test**. Dùng công cụ local/ảnh overlay để giảm thao tác; có thể sinh nhãn gợi ý, nhưng phải được người xem xác nhận trước khi là ground truth.
2. Tách nguồn/người/cảnh để tránh trùng gần giữa split. Tái dùng ảnh cũ cho regression/validation khi phù hợp; ghi thiếu ảnh test thật nếu chưa đủ. Không coi 54 ảnh hiện có tự động đáp ứng phân bổ 80 ảnh của nhóm.
3. Tạo `evaluation/m2_metrics.py` và runner M2 riêng, ví dụ `evaluation/run_m2_eval.py`, để tránh sửa chồng runner chung với M4. Có adapter manifest mới; giữ runner cũ hoạt động.
4. Ghép một target với tối đa một detection **đúng class và bbox IoU ≥0,5** theo ánh xạ đã khóa; chọn IoU lớn nhất, tie-break xác định. Các target không ghép được vẫn nằm mẫu số. Không tính có bất kỳ box nào là đúng target.
5. Tính target detection recall, mask coverage/fallback, IoU raw/morph, latency M2 và kích thước payload. Báo số mẫu/denominator rõ; warm-up tách riêng.
6. Ground truth và prediction phải cùng canvas: lưu mask nhãn trong ảnh chuẩn hóa hoặc lưu bbox/crop transform đầy đủ; đặt predicted ROI mask về canvas chung trước khi tính IoU. Không resize mask GT vào bbox dự đoán rồi bỏ qua sai lệch detection.
7. Báo riêng IoU có điều kiện trên ca ghép đúng/có mask và IoU end-to-end với ca thiếu detection/mask tính 0 theo policy công bố. Thêm phân tích mask trên bbox GT cố định để tách lỗi segmentation khỏi lỗi detector.
8. Test metric bằng hình học biết trước: IoU 0/1/một phần, sai class, box sai món, không detection, mask lệch tọa độ, fallback, tránh chia 0. Lưu schema/example, thông tin split và config cùng CSV/JSON.

**Đầu ra:** manifest, nhãn/mask có provenance, runner offline, metric helper/tests; ảnh overlay nhãn để kiểm chéo. Người 2 kiểm nhãn Người 3; Người 1 kiểm phần Người 2 theo vòng bàn giao, khi có dữ liệu.

**Nghiệm thu:** test metric qua; số liệu tổng hợp truy ngược từng target; không loại ca thất bại khỏi mẫu số. Nếu người dùng chưa xác nhận nhãn thật, ghi phần gán nhãn chờ xác nhận, không thay bằng nhãn tổng hợp.

## Phase 5 — Thực nghiệm và tích hợp có kiểm soát

**Lệnh:** `làm phase 5`

1. Nối M1 thật khi có; kiểm tra hai nhánh, EXIF/resize và bbox bằng fixture chung. Nối M3 decoder/source ROI/mask với fixture PNG ở phase 2–3. M3 sở hữu descriptor; không tự thay descriptor/ranking để có điểm đẹp.
2. Dựng chuỗi so sánh tách yếu tố: **B0** pipeline cũ chạy lại; **R0** ROI nguồn không mask; **B2** cùng R0 + raw mask; **B3** cùng R0 + mask morphology. R0 là control thêm để không gán lợi ích bỏ padding/JPEG cho mask. Báo B0→R0 riêng, B2/B3 so với R0 mới là tác động mask.
3. Với thử nghiệm M2, đóng băng model/bbox/crop variants/embedding/candidate pool/descriptor/ranking. Tính B2/B3 trên cùng ROI của mỗi ảnh. Dùng cùng policy fallback trên query và catalog; M3 quản lý cache version riêng nếu cần. Không ghi đè vector gốc hoặc đổi index giữa các cấu hình.
4. Chọn polarity/gate/morphology chỉ trên validation. So raw/morph/off; khóa config và snapshot trước test. Chỉ chạy test cuối sau khóa, không tune tiếp từ lỗi test rồi gọi đó là test độc lập.
5. Đo mask IoU/coverage/fallback/latency; phối hợp M3 đo màu và tác động retrieval cùng target. Không có M3/DB thì vẫn chạy phần mask offline; bảng màu/retrieval ghi **chưa đo**, có script/fixture và yêu cầu đầu vào cụ thể. Phase 5 chưa hoàn thành phần tích hợp thực nếu mới có mock.
6. Có bảng từng ca + tổng hợp, hình tốt/xấu (nền sáng, nền phức tạp, trắng trên trắng, nhiều lớp/da); lưu `config_id`, commit/model hash, split/catalog hash, phần cứng, cold/warm. Không khẳng định mask phân biệt được da trong mọi tình huống.
7. Chọn mặc định bằng bằng chứng validation và giới hạn rủi ro; nếu không hơn control hoặc lỗi nghiêm trọng, để `off`/fallback làm mặc định, vẫn giữ thuật toán và kết quả phục vụ báo cáo.
8. Chạy test cũ và mới liên quan, HTTP contract/API smoke và detection thật khi model đã có. Ghi lỗi dịch vụ/dữ liệu cùng mẫu số, không bỏ query âm thầm.

**Đầu ra:** bảng B0/R0/B2/B3, cấu hình đã khóa, CSV/JSON, overlay, ghi chú bàn giao M1/M3; kết luận có/không bật morphology/mask.

**Nghiệm thu:** có bằng chứng chạy thật cho phần được kết luận; IoU từ mask nhãn người kiểm tra; không đồng thời thay descriptor/ranking; không gọi detection_success_rate là mAP/recall.

## Phase 6 — Gói cá nhân để ghép bài nộp

**Lệnh:** `làm phase 6`

1. Viết `docs/M2_REPORT.md`, khoảng một trang nội dung cho báo cáo chung: vấn đề, YOLO/ROI, Otsu, morphology/fallback, thí nghiệm, kết quả thực, giới hạn và phân công. Chỉ đưa số liệu đã đo, ghi mẫu số và nhóm dữ liệu.
2. Viết nội dung hai slide trong `docs/M2_SLIDES.md`: pipeline/ảnh mask và bảng đối chứng/ca lỗi; kèm ghi chú thuyết trình khoảng 1,5–2 phút và câu trả lời về Otsu, opening/closing, fallback, IoU.
3. Gắn phần Otsu/phân vùng với chương 5 đã cung cấp. Bàn giao dẫn chương 7 cho morphology nhưng bộ PDF hiện có chỉ đến chương 6; ghi thiếu nguồn chương 7, bổ sung tài liệu học phần khi nhận được hoặc trích nguồn học thuật/official phù hợp lúc viết. Không mô tả YOLO/morphology như thể chúng có đầy đủ trong slide chương 5.
4. Tạo `docs/M2_HANDOFF.md`: file đổi, contract, môi trường, lệnh test/eval thực sự chạy được, model/data cần có, trạng thái integration, cấu hình cuối, checklist chạy lại và khai báo AI. Không đưa lệnh/script dự kiến chưa tồn tại vào README như đã chạy.
5. Gom mẫu 5–10 ảnh phù hợp với gói nhóm, mask/overlay tiêu biểu, config, nhãn và số liệu của M2 vào thư mục bàn giao riêng. Kiểm tra `.gitignore` đang bỏ `*.json`, `models/`, `evaluation/results/`; đưa artifact cần tái lập vào gói hoặc whitelist hẹp theo file, không mở ignore toàn bộ dữ liệu.
6. Chạy lại theo hướng dẫn từ checkout/thư mục tạm sạch hoặc môi trường sẵn có được ghi rõ; không xóa thay đổi hiện tại. Kiểm tra đường dẫn tương đối, không có secrets, có hướng dẫn lấy model hợp lệ. Phân biệt chạy mô phỏng và demo YOLO thật.
7. Cập nhật progress cuối; liệt kê hạng mục còn phụ thuộc nhóm. M2 hoàn tất khi code/tests/evidence/report/handoff đều đủ, không yêu cầu tự viết báo cáo/slide cả nhóm.

**Đầu ra:** ba tài liệu M2, hình và bảng kết quả, gói bàn giao cá nhân có hướng dẫn tái lập. Nếu cần xuất PDF/PPTX riêng, thực hiện khi người dùng yêu cầu định dạng đó; bản Markdown trước giúp ghép báo cáo nhóm nhanh.

## Hợp đồng mặc định cần cụ thể hóa ở phase 1

- M1→M2: `source_rgb`, `detection_rgb` uint8 RGB cùng shape; geometry metadata và nguồn mode. Không gamma source, không EXIF hai lần.
- M2→M3: giữ `image_base64` và `crop_variants` cũ; thêm `schema_version`, `query_id`, `garment_id`, `source_crop_png_base64`, `mask_png_base64`, `roi_metadata`.
- `roi_metadata` tối thiểu: bbox tight, working image size, source ROI width/height, coordinate space, color space, roi_source, roi_status, fallback_reason, method/config version. Metadata legacy của crop padded vẫn giữ ý nghĩa cũ.
- PNG dùng raw base64, không tiền tố data URL. ROI decode RGB Hroi×Wroi×3; mask grayscale Hroi×Wroi, giá trị 0/255, foreground=255. Null mask nghĩa dùng toàn ROI nguồn cho màu và ghi fallback, không nghĩa không có detection.
- Mask chỉ lọc pixel nhánh màu; không thay ảnh embedding. Legacy crop BGR/OpenCV và JPEG vẫn có conversion phù hợp; test đỏ/xanh chứng minh đúng biên.
- Bbox dùng ảnh sau chuẩn hóa M1, x2/y2 exclusive. Metadata đủ chuyển GT từ ảnh upload về working canvas; nếu thiếu transform thì evaluator phải báo không đánh giá được thay vì ngầm so tọa độ khác nhau.

## Phối hợp tối thiểu và xử lý thiếu đầu vào

| Đầu vào | Cần từ ai | Nếu chưa có |
|---|---|---|
| best.pt đúng model quần áo | Đã có `object_detection/best.pt` | Phase 1 nối đường dẫn runtime và kiểm tra load/classes; nếu load lỗi, tiếp tục fixture và ghi lỗi thật |
| PreparedImage và geometry | Người 1 | Adapter identity có ghi mode và fixture hai nhánh |
| Decoder + màu dùng ROI/mask | Người 3 | Bàn giao PNG/round-trip, chạy IoU offline; màu/retrieval để chưa đo |
| Catalog/candidate snapshot | Người 3/4 | Giữ thí nghiệm segmentation độc lập, chưa kết luận retrieval |
| Nhãn và kiểm chéo | Người dùng/nhóm | Tạo công cụ/overlay, người xem xác nhận; không gọi pseudo-label là GT |
| Tổng ≥50 ảnh đánh giá theo đề | Cả nhóm | Người 2 góp 20 ảnh; tập chung theo bàn giao 20 val + 60 test sau khử trùng |

Chỉ yêu cầu người dùng cung cấp đầu vào không thể tự lấy hợp lệ hoặc xác nhận nhãn cần mắt người. Gom yêu cầu ngắn gọn; không lặp lại xin phép cho các chỉnh sửa nằm trong phase đã được yêu cầu.

## Thứ tự cắt phạm vi nếu sát hạn

1. Bỏ mọi mở rộng ngoài Otsu và kernel nhỏ; không thêm UI demo mới.
2. Giữ test hình học/màu/HTTP, fallback và metric đúng; không đổi chúng lấy thêm thuật toán.
3. Giữ phần dữ liệu/nhãn thật, đối chứng và ca thất bại; báo rõ cỡ mẫu nhỏ nếu không đạt phân bổ, không thay bằng số liệu giả.
4. Khi integration nhóm chậm, bàn giao module độc lập + fixture + số liệu offline trước; phần chưa đo đánh dấu rõ, không nhận làm thay M3/M4.
5. Ngày 10/11 ưu tiên sửa lỗi tái lập/đóng gói; không mở hướng nghiên cứu mới.
