# Phân công 4 người theo module trong luồng xử lý

Cập nhật: 28/10/2026. Bản này thay cách chia theo nhóm cải tiến trước đó.

## Luồng và quyền sở hữu

Ảnh → M1. Tiền xử lý → M2. Phát hiện/ROI/mask → M3. Trích đặc trưng → M4. Truy xuất/xếp hạng → Kết quả.

Mỗi người sở hữu một module trong luồng sản phẩm, kèm kiểm thử, thực nghiệm và phần báo cáo tương ứng. Đánh giá và dữ liệu là phần chung được chia đều.

## M1 — Người 1: Chuẩn hóa và tiền xử lý ảnh

**Giá trị:** đầu vào nhất quán, thử tăng độ bền trước ảnh tối mà vẫn giữ màu gốc.

- Chuẩn hóa hướng EXIF, chế độ RGB và kích thước làm việc; giữ tỉ lệ khung hình.
- Giữ ảnh nguồn sau chuẩn hóa và bản riêng phục vụ detection, cùng kích thước/hệ tọa độ.
- Cài đánh giá độ sáng và gamma có điều kiện; chọn tham số trên validation.
- Giữ metadata biến đổi để quy đổi bbox về ảnh upload; gamma không thay đổi hình học.
- Kiểm thử ảnh xoay, RGBA, ảnh tối/đủ sáng, các kích thước khác nhau.
- Đo thời gian tiền xử lý và tác động end-to-end riêng trên ảnh tối/đủ sáng.

**Đầu vào:** ảnh giải mã và cấu hình.
**Đầu ra dự kiến:** source_rgb, detection_rgb, geometry_meta, quality_meta.

**Vùng code:** module mới AI/image_preprocessing.py; điểm gọi decode/tiền xử lý trong object_detection/detector.py.

**Nghiệm thu:** M2 dùng được đầu ra; không ghi đè màu gốc bằng ảnh tăng sáng; kiểm thử hình học qua; có bảng gamma bật/tắt và latency.

**Giới hạn:** chưa thêm CLAHE, nhiều bộ lọc hoặc tối ưu JPEG trong gói bắt buộc. Không thay mô hình.

## M2 — Người 2: Phát hiện, ROI và mask

**Giá trị:** định vị món đồ và giảm nền/da/món khác trong vùng lấy đặc trưng.

- Chạy YOLO hiện có trên detection_rgb; chuẩn hóa bbox, clip biên, thử một tập nhỏ ngưỡng confidence.
- Cắt ROI từ source_rgb cùng hệ tọa độ; giữ crop gốc và thông tin padding.
- Cài một phương pháp mask cổ điển có phạm vi rõ, như Otsu hoặc phát triển vùng cho nền tương đối đồng nhất.
- So sánh mask thô với opening/closing nhẹ; fallback khi mask rỗng/bất thường.
- Bàn giao mask cùng kích thước ROI, bbox, class và confidence.
- Đánh giá ghép bbox/IoU, mask IoU trên tập có nhãn và tỷ lệ fallback.

**Đầu vào:** hai nhánh ảnh và metadata của M1.
**Đầu ra dự kiến:** danh sách GarmentROI gồm id, bbox, class, confidence, RGB crop gốc, mask tùy chọn, metadata padding.

**Vùng code:** phần YOLO/ROI của object_detection/detector.py; module mới AI/foreground.py; schema/serialization detection khi cần.

**Nghiệm thu:** M3 nhận ROI/mask đúng tọa độ; có fallback, kết quả detection/mask và ca thất bại. Dùng mask cho màu trước.

**Giới hạn:** chỉ một phương pháp mask và morphology; không huấn luyện lại YOLO, không xây segmentation tổng quát. Mask hình học hợp lệ không có nghĩa là tách đúng quần áo.

## M3 — Người 3: Trích xuất đặc trưng

**Giá trị:** biểu diễn món đồ để so sánh, khắc phục lỗi màu trắng và hạn chế của một RGB trội.

- Giữ FashionCLIP hiện có, preprocessing đúng yêu cầu encoder; kiểm tra vector 512 chiều và chuẩn hóa.
- Sửa cách lấy màu, phân biệt padding đã biết với vải trắng/kem; dùng mask nếu có, ROI fallback nếu không.
- Cài histogram HSV với nhánh trắng/xám/đen; giữ descriptor cũ để đối chứng.
- Dùng cùng hàm tạo đặc trưng cho query/catalog; gắn phiên bản và ghi cache thử nghiệm riêng.
- Cung cấp embedding và histogram cho M4; histogram dùng được cho đối chứng cổ điển.
- Kiểm thử màu trắng, nhiều màu, nhóm vô sắc, mask rỗng và vector sai chiều; đo độ chính xác màu và thời gian trích đặc trưng.

**Đầu vào:** GarmentROI từ M2, ảnh catalog theo cùng quy tắc ROI đã chốt.
**Đầu ra dự kiến:** embedding, descriptor màu, tên màu hiển thị, class và metadata phiên bản.

**Vùng code:** AI/color_utils.py, AI/embedding_model.py, phần tạo đặc trưng trong AI/vectorizer.py; module descriptor/cache mới khi cần.

**Nghiệm thu:** M4 dùng được đặc trưng; catalog/query cùng quy tắc; có kiểm thử và kết quả màu cũ/mới. Không dùng tên màu do thuật toán sinh làm ground truth.

**Giới hạn:** không fine-tune FashionCLIP, không thêm nhiều họ descriptor/HOG, không tự viết ranking hoặc đổi schema DB.

## M4 — Người 4: Truy xuất, xếp hạng và quyết định kết quả

**Giá trị:** tìm và sắp xếp sản phẩm phù hợp; xác định khi nào chưa đủ tin cậy.

- Giữ pgvector hiện có; kiểm tra phiên bản đặc trưng và số lượng ứng viên.
- Dùng descriptor M3 để tính điểm màu, kết hợp điểm ảnh/loại; xuất score breakdown.
- Chọn trọng số và ngưỡng trên validation; xử lý không có ứng viên và kết quả yếu.
- Cài ba chế độ: histogram-only cổ điển, embedding-only và hybrid trên cùng giao thức đầu vào.
- Ghép điểm gọi M3/M4 trong api/search_service.py; giữ response tương thích frontend.
- Đo HitRate@5, MRR ở độ sâu công bố, đúng loại/màu trên cùng kết quả và thời gian retrieval/rerank; phân biệt lỗi thiếu ứng viên với sai thứ hạng.

**Đầu vào:** query features và catalog/index/cache tương thích.
**Đầu ra dự kiến:** top-K sản phẩm, các thành phần điểm, trạng thái đủ/chưa đủ tin cậy và metadata truy xuất.

**Vùng code:** api/search_service.py; hàm truy xuất liên quan trong db/database.py; runner so sánh chế độ khi cần.

**Nghiệm thu:** ba chế độ chạy lại được; API trả đúng schema; có bảng chất lượng/thời gian, ca từ chối và kiểm thử bằng features giả lập.

**Giới hạn:** không viết lại DB/ETL/frontend, không gánh toàn bộ evaluator hoặc tích hợp hệ thống. Dùng cache descriptor có phiên bản cho thử nghiệm thay vì migration lớn.

## Đánh giá và dữ liệu chia đều

Không giao riêng một người chỉ làm benchmark. Mỗi người tự chạy và phân tích thí nghiệm module mình; chạy lại end-to-end và ghép bảng tổng hợp thực hiện chung.

| Người | Phần hỗ trợ evaluator |
|---|---|
| 1 | Manifest/split và đo thời gian, phân nhóm đủ sáng/thiếu sáng |
| 2 | Ghép đúng bbox mục tiêu, kiểm thử IoU và không dùng lẫn detection |
| 3 | Kiểm tra nhãn màu độc lập, độ chính xác màu, hỗ trợ joint loại/màu |
| 4 | HitRate/MRR, quy tắc không có sản phẩm phù hợp và kết quả bị từ chối |

Sau khi sửa metric, đo lại pipeline gốc làm mốc. Không so trực tiếp với số cũ dùng định nghĩa khác.

Mỗi người chuẩn bị 20 ảnh: 5 validation + 15 test; cả nhóm có 20 validation + 60 test sau kiểm tra trùng/nguồn ảnh. Mỗi ảnh gán một món mục tiêu, bbox/loại/màu/relevance theo schema chung. Trong số đó, mỗi người gán thêm 5 mask tham chiếu (2 validation + 3 test), tổng 20 mask cho đánh giá phụ trợ.

Kiểm nhãn theo vòng 1 → 2 → 3 → 4 → 1. Không dùng nhãn màu do thuật toán sinh làm chuẩn. Mỗi người viết khoảng 1 trang phương pháp/kết quả, 2 slide và ca đúng/sai.

## Cân bằng công sức

Giả định kinh nghiệm Python tương đương, dự trù mỗi người 40 giờ:

| Hạng mục | Giờ/người |
|---|---:|
| Đọc tài liệu, thiết kế, interface | 4 |
| Cài đặt module | 12 |
| Kiểm thử và tích hợp | 6 |
| Thực nghiệm và phân tích | 8 |
| Dữ liệu và kiểm nhãn | 6 |
| Báo cáo, slide | 4 |
| **Tổng** | **40** |

Cân bằng bằng phạm vi và thời gian thực tế, không bằng số module hoặc số hàm. Sau hai ngày, đối chiếu giờ và việc còn lại. Nếu mask quá khó, giới hạn điều kiện áp dụng thay vì biến M2 thành dự án segmentation riêng. Nếu gói nhẹ hơn dự kiến, nhận thêm kiểm thử/đối chứng thuộc ranh giới module.

Không nghiệm thu bằng cam kết tăng một tỷ lệ metric. Module đúng, thử công bằng và phân tích được thất bại vẫn là đóng góp hợp lệ.

## Interface, làm song song và tích hợp

- Chốt DTO ngày đầu; dùng fixtures/mock để chạy độc lập, không chờ công đoạn trước.
- M1 và M2 cùng liên quan detector.py: M1 đặt logic tiền xử lý trong module riêng; M2 sở hữu detect/ROI. Ghép điểm gọi theo thứ tự.
- M3 trả features qua interface; M4 sở hữu điểm gọi trong search_service.py.
- Mỗi chủ module cùng chủ module kế tiếp kiểm thử bàn giao. Không có một người tích hợp hộ cả nhóm.
- Catalog dùng cùng quy tắc ROI/đặc trưng phù hợp từ M1–M3 rồi M4 lập index/cache. Chốt fallback và lưu metadata khi không có detection.
- Giữ nguyên nhãn test và cấu hình/cache/kết quả gốc; không chọn tham số theo test.

## Bộ thí nghiệm

1. Pipeline gốc đo lại bằng evaluator mới.
2. Chỉ thay M1: gamma có điều kiện, đánh giá ảnh tối/đủ sáng.
3. Chỉ thay M2: mask thô, sau đó thêm morphology để tách tác dụng.
4. Chỉ thay M3: sửa màu trắng, sau đó HSV; giữ bbox/tập ứng viên cố định khi đo descriptor.
5. Chỉ thay M4: histogram-only, embedding-only, hybrid với cùng ROI/relevance.
6. Kết hợp các cấu hình chọn trên validation; đo end-to-end trên test đã khóa.

Giữ catalog, top-K và ngân sách ứng viên cố định cho phép so sánh tương ứng. Báo cáo riêng thay đổi số crop/ứng viên. Metric chính: HitRate@5, MRR ở độ sâu công bố, Joint Category+Color@5, độ chính xác màu, latency. IoU chỉ tính khi có nhãn tương ứng.

## Tiến độ

| Thời gian | Đầu ra |
|---|---|
| 28–29/10 | Interface, fixtures, dữ liệu/split và quy tắc metric |
| 30/10–03/11 | Bốn module, kiểm thử độc lập và validation |
| 04–06/11 | Ghép cặp module, khóa cấu hình, chạy test |
| 07–09/11 | Báo cáo/slide và ghép sản phẩm nộp |
| 10/11 | Chạy lại từ README, sửa lỗi đóng gói |
| 11/11 | Nộp theo yêu cầu môn học |

Nếu thiếu thời gian, bỏ phần tùy chọn trước; giữ phép đo đúng, đối chứng và tối thiểu 50 ảnh test.
