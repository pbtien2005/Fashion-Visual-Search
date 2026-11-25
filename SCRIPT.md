# Kịch bản thuyết trình — Fashion Visual Search

Theo **Slide Xử lý ảnh.pdf** (16 slide). Lời thoại dự kiến khoảng 10–12 phút ở tốc độ nói vừa phải. Các đoạn **Lời nói** có thể đọc trực tiếp; **Ghi chú** chỉ để người trình bày tự kiểm tra trước khi lên lớp.

## Slide 1 — Trang bìa

**Lời nói:**

“Em kính chào thầy và các bạn. Nhóm em xin trình bày đồ án Xử lý ảnh với đề tài *Fashion Visual Search*: tìm sản phẩm thời trang tương tự từ ảnh người đang mặc đồ. Trong bài trình bày, em sẽ đi từ bài toán, cách hệ thống xử lý ảnh, đến kết quả và những giới hạn đã quan sát được.”

## Slide 2 — Bạn muốn tìm món nào trong ảnh?

**Lời nói:**

“Khi nhìn một bức ảnh outfit, người dùng có thể thích chiếc áo, chiếc quần hoặc chiếc váy, chứ không nhất thiết muốn tìm một sản phẩm giống cả bức ảnh. Vì thế, hệ thống phải nhận ra từng món trước. Mỗi món được cắt thành một vùng ảnh riêng và nhận một danh sách sản phẩm tương tự riêng. Đây là lý do khâu phát hiện quần áo đứng trước khâu tìm kiếm.”

## Slide 3 — Hai luồng của hệ thống

**Lời nói:**

“Hệ thống có hai luồng. Luồng ngoại tuyến chuẩn bị catalog: thu thập thông tin và ảnh sản phẩm, trích đặc trưng, rồi lưu để truy vấn. Luồng trực tuyến bắt đầu khi người dùng tải ảnh lên: M1 chuẩn hóa ảnh; M2 phát hiện và cắt từng món; M3 tạo đặc trưng ảnh và màu; M4 truy vấn catalog và xếp hạng kết quả. Tách hai luồng như vậy giúp đặc trưng của sản phẩm được tính trước, còn lúc người dùng tìm kiếm chỉ cần xử lý ảnh truy vấn và so với dữ liệu đã chuẩn bị.”

## Slide 4 — Từ Amazon đến kho sản phẩm

**Lời nói:**

“Ở nhánh ngoại tuyến, nhóm tìm sản phẩm trên Amazon US theo từ khóa, dùng ASIN để nhận diện và loại trùng. Ảnh được lưu qua Cloudflare R2, còn metadata gồm tên, loại, giá và đường dẫn sản phẩm. Sau đó hệ thống tính vector ảnh FashionCLIP và descriptor màu HSV. Vector ảnh cùng metadata được lưu vào PostgreSQL có pgvector; chỉ mục HNSW hỗ trợ tìm ứng viên theo cosine. Descriptor màu được giữ trong cache theo ID sản phẩm để dùng khi xếp hạng.”

“Con số 1.752 trên slide là quy mô catalog của **lượt benchmark lịch sử**. Với demo hybrid được lưu trong repository hiện tại, nhóm dùng catalog mẫu 28 sản phẩm; em sẽ nói rõ khi đến slide demo.”

**Ghi chú:** Không gọi 1.752 là số sản phẩm đang chạy trong demo 28 ảnh. Repo hiện có `data/fashion_metadata.json` và `data/fashion_with_hsv_vectors.json` mỗi file 28 mục; chưa có artifact ở đây để kiểm lại catalog benchmark 1.752 mục.

## Slide 5 — 75 ảnh ngoài đời để đánh giá

**Lời nói:**

“Để đánh giá gần tình huống người dùng hơn, bộ benchmark lịch sử dùng 75 ảnh ngoài đời từ Wikimedia Commons. Mỗi ảnh có một món mục tiêu; dữ liệu được chia thành 23 ảnh validation để chọn cấu hình và 52 ảnh test để đánh giá sau khi khóa cấu hình. Phần nhãn mô tả khung và loại quần áo, màu, điều kiện sáng, cùng sản phẩm được xem là phù hợp. Việc chia tập nhằm tránh điều chỉnh tham số theo chính tập dùng để báo cáo kết quả.”

**Ghi chú:** Những con số 75, 23 và 52 là nội dung slide của benchmark lịch sử. `evaluation/labels.real.json` trong repo hiện tại chỉ chứa 16 truy vấn, còn tài liệu M2 dùng 20 target chung từ M1. Không trình bày các tập này như cùng một bộ dữ liệu.

## Slide 6 — M1: Chuẩn hóa ảnh đầu vào

**Lời nói:**

“M1 xử lý các khác biệt của ảnh tải lên trước khi phát hiện. Đầu tiên là sửa chiều ảnh theo EXIF, sau đó chuyển về RGB; ảnh trong suốt được ghép lên nền trắng. Nếu ảnh lớn, M1 thu nhỏ bằng nội suy Lanczos để cạnh dài tối đa 1.280 pixel mà vẫn giữ tỷ lệ. M1 còn có phương án đo độ sáng và làm sáng ảnh tối bằng gamma có điều kiện, nhưng cấu hình mặc định hiện tại là `identity`, tức giữ độ sáng gốc sau các bước chuẩn hóa.”

“Kết quả 20 trên 47 ở slide là HitRate@5 của phép đo benchmark trước khi áp ngưỡng chấp nhận. Hai cấu hình có và không dùng gamma đạt cùng con số này; điều đó chưa đủ để nói gamma cải thiện truy hồi.”

## Slide 7 — M2: Phát hiện và cắt từng món

**Lời nói:**

“Từ ảnh đã chuẩn hóa, M2 dùng YOLOv8m để phát hiện 10 lớp quần áo, với ngưỡng tin cậy 0,5. Mỗi dự đoán hợp lệ có loại trang phục và bounding box. Từ một khung, M2 tạo ba biến thể: khung sát, nới 15 phần trăm và nới 30 phần trăm. Mỗi biến thể được đưa về ảnh 224 nhân 224 để FashionCLIP mã hóa. Mục đích của nhiều biến thể là cho bộ truy hồi thêm lựa chọn về lượng ngữ cảnh quanh món đồ.”

“Ngoài ba ảnh cho encoder, M2 còn giữ một ROI màu cắt sát theo bounding box từ ảnh nguồn. Vùng màu này không lấy từ ảnh đã thêm viền trắng hoặc tăng gamma, để tránh làm sai lệch histogram.”

## Slide 8 — M2: Thử tách nền bằng Otsu

**Lời nói:**

“Bounding box vẫn có thể chứa một phần nền. Nhóm đã thử tạo mask để chỉ lấy pixel quần áo khi tính màu. Thuật toán Otsu tự chọn ngưỡng chia vùng sáng và tối trong ROI. Sau đó, hệ thống chọn phía có xu hướng nằm giữa ảnh hơn là ở viền; phép mở rồi đóng với kernel 3 nhân 3 có thể loại đốm nhỏ và lấp lỗ nhỏ. Trước khi chấp nhận mask, chương trình còn kiểm tra độ tương phản, diện tích và tính ổn định.”

“Ví dụ trên slide, IoU của mask Otsu là 0,719 và sau mở–đóng là 0,726, vẫn thấp hơn ngưỡng yêu cầu 0,8. Vì phương án này chưa đạt tiêu chí trên mọi ảnh tham chiếu, cấu hình chốt để **mask tắt mặc định**. Hệ thống hiện dùng toàn ROI trong khung YOLO để tính màu. Otsu ở đây là phân ngưỡng sáng tối, chưa phải mô hình hiểu ngữ nghĩa quần áo.”

## Slide 9 — M3: Mô tả kiểu dáng và màu

**Lời nói:**

“M3 tạo hai loại đặc trưng. FashionCLIP biến ảnh crop thành vector 512 chiều, dùng để so độ tương đồng về hình ảnh và kiểu dáng. Nhánh màu chuyển ROI sang HSV và tạo histogram 147 chiều: 18 ô sắc độ nhân 8 ô độ bão hòa, cộng ba ô dành riêng cho đen, xám và trắng. Histogram được chuẩn hóa theo tỷ lệ pixel. Nếu mask đang tắt, toàn bộ pixel trong ROI được tính, nên nền nằm trong bounding box vẫn có thể ảnh hưởng kết quả màu.”

“Hai đặc trưng này bổ sung cho nhau: embedding giúp tìm ứng viên giống về hình ảnh; histogram giúp so phân bố màu. Tên màu hiển thị chỉ là nhãn xấp xỉ suy ra từ histogram, không phải đặc trưng dùng để so điểm.”

## Slide 10 — M4: Xếp hạng sản phẩm tương tự

**Lời nói:**

“Ở chế độ hybrid, mỗi biến thể crop được mã hóa thành vector ảnh. pgvector tìm tối đa 100 sản phẩm ứng viên theo cosine, với đúng model và phiên bản embedding. Sau đó M4 tính lại điểm cho từng ứng viên bằng bốn tín hiệu: 60 phần trăm độ giống ảnh, 15 phần trăm khớp loại quần áo, 15 phần trăm độ giống histogram màu và 10 phần trăm văn bản hoặc kiểu. Một số trường hợp sai loại rõ rệt còn bị trừ điểm.”

“Nếu cùng một sản phẩm xuất hiện ở nhiều biến thể crop, hệ thống giữ **điểm tốt nhất của sản phẩm đó**, rồi sắp xếp lấy top K. Với hybrid hiện tại, chỉ sản phẩm có điểm cuối **lớn hơn 0,60** được hiển thị. Đây là quy tắc triển khai ban đầu, chưa phải ngưỡng đã được hiệu chỉnh trên tập ảnh độc lập.”

## Slide 11 — Demo từ ảnh đến kết quả

**Lời nói:**

“Đây là một kết quả API hybrid đã lưu ngày 16 tháng 11. Từ ảnh tải lên, YOLO phát hiện một áo tay ngắn. Hệ thống tìm được sản phẩm B0FRRYXKFC ở hạng nhất với điểm 0,885; hai sản phẩm tiếp theo có điểm 0,727 và 0,719. Các điểm này là điểm hybrid sau khi gộp tín hiệu và chọn biến thể crop tốt nhất, không phải xác suất sản phẩm đúng.”

“Ảnh truy vấn ở ví dụ này lấy từ **chính catalog mẫu 28 sản phẩm**. Vì vậy demo chứng minh luồng từ upload đến kết quả đang hoạt động; nó không đại diện cho độ chính xác trên ảnh outfit mới ngoài đời.”

## Slide 12 — 20/47 truy vấn trúng trong top 5

**Lời nói:**

“Slide này quay lại **benchmark lịch sử trên ảnh ngoài đời**, không phải demo catalog 28 ảnh vừa rồi. Trong 47 truy vấn có sản phẩm phù hợp để đánh giá, 20 truy vấn có sản phẩm phù hợp xuất hiện trong top 5 trước khi áp ngưỡng chấp nhận. Cấu hình B0 là cấu hình benchmark cũ; chênh lệch với phương án dùng embedding của vùng cắt là ba ảnh.”

“Khi đọc con số này, cần nhớ nó chỉ tính những truy vấn có nhãn sản phẩm phù hợp và được đo trước bước lọc ngưỡng. Nó không phải kết quả đánh giá ba chế độ truy hồi mới trên một tập ảnh độc lập.”

**Ghi chú:** Nếu được hỏi B0 dùng đúng công thức HSV147 hybrid hiện tại hay không, trả lời **không**: tài liệu dự án cảnh báo benchmark lịch sử có trước tích hợp HSV production. Không suy 20/47 thành Recall@5 của API hybrid hiện nay.

## Slide 13 — Phát hiện là điểm nghẽn chính

**Lời nói:**

“Ở tập test ngoài đời của benchmark trên slide, chỉ 27 trên 52 món mục tiêu được dự đoán **đúng lớp** và có IoU bounding box ít nhất 0,5. Trong ví dụ Harajuku, vùng áo khoác bị nhận sai loại. Nếu M2 phát hiện sai hoặc cắt lệch, M3 sẽ mã hóa nhầm vùng ảnh; lúc đó M4 dù xếp hạng tốt cũng khó tìm đúng sản phẩm. Vì vậy, cải thiện phát hiện trên ảnh người mặc là hướng ưu tiên rõ nhất.”

“Con số 27/52 đánh giá bước phát hiện theo nhãn của benchmark đó; nó không phải độ chính xác sản phẩm của toàn hệ thống.”

## Slide 14 — Ngưỡng làm thay đổi kết quả

**Lời nói:**

“Trong phép đo B0 lịch sử, trước lọc ngưỡng có 20 trên 47 truy vấn trúng top 5, nhưng sau ngưỡng benchmark ban đầu thì còn 0 trên 47. Ngưỡng khi đó yêu cầu điểm tổng ít nhất 0,70 và độ giống ảnh ít nhất 0,60. Kết quả này cho thấy bộ lọc có thể loại cả ứng viên phù hợp nếu chưa được hiệu chỉnh trên dữ liệu đại diện.”

“Cần phân biệt với code hybrid hiện tại: hệ thống đang dùng điều kiện **điểm cuối lớn hơn 0,60**, không áp đồng thời hai ngưỡng B0 trên slide. Hai con số trước và sau ngưỡng này thuộc benchmark lịch sử, không phải phép đo lại trên API hybrid hiện tại.”

## Slide 15 — Kết luận và hỏi đáp

**Lời nói:**

“Tóm lại, nhóm đã xây dựng luồng tìm kiếm từ ảnh người mặc theo từng món: M1 chuẩn hóa, M2 phát hiện và cắt ROI, M3 trích embedding cùng màu, M4 truy hồi và xếp hạng. Demo catalog cho thấy luồng kỹ thuật chạy được. Tuy nhiên, kết quả trên ảnh ngoài đời còn phụ thuộc nhiều vào phát hiện đúng món đồ; ngưỡng chấp nhận cũng cần được kiểm chứng lại.”

“Hai bước ưu tiên tiếp theo là cải thiện phát hiện trên ảnh ngoài đời và đánh giá lại ngưỡng trên tập ảnh độc lập có nhãn sản phẩm phù hợp. Em xin kết thúc phần trình bày và sẵn sàng trả lời câu hỏi.”

## Slide 16 — Trang cuối

**Lời nói:**

“Em cảm ơn thầy và các bạn đã lắng nghe.”

**Ghi chú cần sửa trước khi trình chiếu:** Slide bìa ghi **Nhóm 09**, còn slide cuối ghi **Nhóm 13**. Xác nhận số nhóm và sửa một trong hai slide; không đọc số nhóm ở slide cuối khi chưa sửa.

## Câu hỏi dễ gặp

**Otsu có tách chính xác quần áo khỏi nền không?** “Không. Otsu phân pixel theo độ sáng và kiểm tra hình học đơn giản. Vì độ sáng áo và nền có thể giống nhau, nhóm giữ mask tắt trong cấu hình hiện tại.”

**Hybrid có gộp hai danh sách histogram-only và embedding-only không?** “Không. Hybrid dùng embedding để lấy ứng viên trước, rồi tính lại điểm của ứng viên bằng ảnh, loại đồ, HSV và văn bản/kiểu. Sau đó mới gộp kết quả từ các biến thể crop theo điểm tốt nhất của từng sản phẩm.”

**Vì sao demo tốt nhưng benchmark ảnh ngoài đời thấp hơn?** “Ảnh demo thuộc chính catalog, còn ảnh ngoài đời có tư thế, nền, che khuất và ánh sáng khác. Hơn nữa hai kết quả đến từ các phép đo và cấu hình khác thời điểm, nên không so như một phép thử đối chứng.”

**20/47 có phải kết quả hybrid hiện tại không?** “Không. Đó là benchmark B0 lịch sử trước ngưỡng. API hybrid hiện tại đã có smoke test 28/28 ảnh từ chính catalog, nhưng chưa có phép đo tương đương trên bộ outfit độc lập có nhãn phù hợp.”

## Nguồn đối chiếu trong repository

- `object_detection/image_preprocessing.py`, `object_detection/detector.py`, `object_detection/segmentation.py` — M1 và M2.
- `AI/hsv_descriptor.py`, `AI/embedding_model.py`, `api/search_service.py`, `db/database.py` — M3, M4 và chỉ mục vector.
- `docs/M2_CONTRACT.md`, `docs/M4_HANDOFF.md`, `docs/M4_REVIEW.md`, `docs/EVALUATION.md` — cấu hình, giới hạn và cách hiểu chỉ số.
- `evaluation/RETRIEVAL_MODES_SMOKE.md`, `evaluation/results/m4_hybrid_full_catalog.json` — demo và smoke test ngày 16/11/2026.
