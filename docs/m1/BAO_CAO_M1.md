# M1 · Tiền xử lý ảnh truy vấn

**Fashion Visual Search · Xử lý ảnh**

Ảnh điện thoại có thể chứa EXIF làm thay đổi hướng hiển thị, có kích thước lớn hoặc thiếu sáng. M1 chuẩn hóa ảnh tải lên trước YOLO để bbox và crop dùng cùng một hệ tọa độ. Pillow lấy khung đầu, áp hướng EXIF, chuẩn hóa RGB uint8 và ghép alpha lên nền trắng. Ảnh vượt 50 triệu pixel bị từ chối. Khi cạnh dài vượt 1.280 px, Lanczos lấy mẫu lại theo tỷ lệ và làm tròn mỗi chiều. Upload hỏng trả HTTP 400 khi dịch vụ có model.

`prepare_image(bytes, config)` trả `PreparedImage` phiên bản 1. `source_rgb` giữ pixel chuẩn hóa cho crop và lấy màu. `detection_rgb` dùng vùng nhớ riêng để YOLO có thể nhận ảnh đã tăng sáng. M1 đo luma bằng `(0,299R + 0,587G + 0,114B) / 255` và ghi trung bình, trung vị, p5 và p95. B1 dùng bảng tra 256 mức theo `s = r^γ` khi thống kê được chọn nhỏ hơn ngưỡng. B0 (`identity`) chỉ chạy bước chuẩn hóa. Detector đổi RGB sang BGR khi đưa hai nhánh vào OpenCV và YOLO.

## Kết quả thực nghiệm

Validation khóa ngưỡng mean luma **0,30** và gamma **0,7**. Cả B0 và B1 khớp 3/3 ảnh tối và 1/2 ảnh đủ sáng trên validation. B1 không đạt điều kiện cải thiện số ảnh tối khớp. Với gamma mạnh hơn, ảnh đầm đen của Canadian Film Centre có IoU đúng class giảm từ 0,63 ở B0 xuống 0 ở ngưỡng 0,30 và gamma 0,4. [Ảnh so sánh có bbox](../../evaluation/results/m1_b1_20261109/validation/B1-01-loss-01.jpg) · [Ảnh gốc](https://commons.wikimedia.org/wiki/File:%27An_Evening_with_Judd_Apatow_%26_Leslie_Mann%27_%288224252457%29.jpg) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

| Nhóm test | Chế độ | Ảnh khớp | Số detection | Bật gamma | Lần đầu (ms) | TB / p95 ổn định (ms) |
|---|---|---:|---:|---:|---:|---:|
| Tối | B0 | 3/5 | 18 | 0% | 66,2 | 37,1 / 57,3 |
| Tối | B1 | 3/5 | 16 | 40% | 21,4 | 42,0 / 67,5 |
| Đủ sáng | B0 | 8/10 | 39 | 0% | — | 35,0 / 64,2 |
| Đủ sáng | B1 | 8/10 | 39 | 20% | — | 37,2 / 66,9 |

**Giữ `identity` mặc định.** B1 giữ nguyên 11/15 ảnh khớp và làm giảm hai detection ngoài món mục tiêu ở một ảnh tối. Với ảnh quần đen của Stefan Brending, B1 bật gamma nhưng giữ kết quả khớp ở IoU 0,71. Lần gọi đầu ở mỗi cấu hình dùng ảnh thuộc nhóm tối nên bảng chỉ ghi thời gian lần đầu ở nhóm đó. Kết quả đo trên Windows CPU, PyTorch 2.14.0 và Ultralytics 8.4.208. [Ví dụ giữ được kết quả](../../evaluation/results/m1_b1_20261109/test/B1-retained-01.jpg) · [Ảnh gốc](https://commons.wikimedia.org/wiki/File:2015_RiP_Bastille_Dan_Smith_by_2eight_-_3SC5305.jpg) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

[Cấu hình khóa](../../evaluation/results/m1_b1_20261109/locked_config.json) · [Validation CSV](../../evaluation/results/m1_b1_20261109/validation/summary.csv) · [Validation JSON](../../evaluation/results/m1_b1_20261109/validation/results.json) · [Test CSV](../../evaluation/results/m1_b1_20261109/test/summary.csv) · [Test JSON](../../evaluation/results/m1_b1_20261109/test/results.json)

## Bàn giao cho Người 2

- Bbox từ YOLO nằm trong tọa độ **sau EXIF và resize**, dùng trực tiếp cho cả hai nhánh. Để quy đổi về ảnh đã sửa hướng, nhân tọa độ x với `oriented_width / width` và y với `oriented_height / height`. Sau đó đảo phép biến đổi EXIF nếu cần tọa độ file gốc. `scale = width / oriented_width` mô tả tỷ lệ trục x.
- Đề xuất dùng `_process(image, detection_image=None)`. `image` là nguồn crop BGR, `detection_image` là đầu vào YOLO BGR. Khi bỏ đối số thứ hai, `_process` dùng `image` cho cả hai. Metadata hiện nằm trong `PreparedImage` và log mức DEBUG. Người 2 cần duyệt vị trí metadata nếu nhóm muốn đưa lên response.
- Các biến cấu hình gồm `PREPROCESS_MODE`, `PREPROCESS_MAX_SIDE`, `PREPROCESS_MAX_PIXELS`, `PREPROCESS_BRIGHTNESS_STAT`, `PREPROCESS_DARK_THRESHOLD` và `PREPROCESS_GAMMA`. Giá trị khởi đầu lần lượt là `identity`, `1280`, `50000000`, `mean_luma`, `0.35` và `0.6`. Bộ tham số B1 trong thực nghiệm nằm trong `locked_config.json`.
- Mã xử lý nằm trong `object_detection/image_preprocessing.py`. Adapter nằm trong `object_detection/detector.py`, còn `object_detection/api.py` ánh xạ lỗi decode. Bộ test nằm trong `tests/test_image_preprocessing.py`. Fixture và metadata nằm trong `tests/fixtures/preprocessing/`. Fixture để `preprocess_ms` ở `null` để có thể tái tạo. Dữ liệu và runner nằm trong `evaluation/`.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m tests.fixtures.preprocessing.generate
```

## Giao thức B1

Bộ đánh giá gồm 5 ảnh validation và 15 ảnh test. Mỗi ảnh có một món mục tiêu. Bbox `xyxy` được gán thủ công trên ảnh đã định hướng, theo tỷ lệ 0–1. Nhóm `dark` và `normal` được chọn theo độ rõ của món đồ và ánh sáng hiện trường trước khi xem dự đoán, độc lập với ngưỡng luma của thuật toán. CSV lưu thêm màu, nguồn, tác giả và giấy phép. Dữ liệu gồm ảnh outfit từ repo và 5 ảnh bổ sung từ Wikimedia Commons. Nhãn bbox và nhóm sáng có sai số chủ quan, cỡ mẫu còn nhỏ.

Hai chế độ dùng cùng `best.pt`, confidence 0,5 và cấu hình crop. Validation thử 12 cặp từ ba ngưỡng (0,30, 0,35 và 0,40) và bốn mức gamma (0,4, 0,5, 0,6 và 0,7). Xếp ưu tiên theo số ca đủ sáng bị mất, số ca khớp tăng ở nhóm tối, rồi số ảnh bị bật gamma. Số detection đếm các mục detector trả về sau bước tạo crop. Mỗi ảnh khớp khi có detection đúng class với IoU ≥ 0,50. Runner khóa cấu hình trước khi chạy test, từ chối ảnh trùng validation và dành riêng thư mục cho mỗi lượt chạy. Bbox xanh là nhãn, bbox cam là dự đoán.

```powershell
python -m evaluation.run_m1_ablation validation --labels evaluation/m1_validation.csv --model models/object-detection/best.pt --run-dir evaluation/results/m1_new_run
python -m evaluation.run_m1_ablation test --labels evaluation/m1_test.csv --model models/object-detection/best.pt --run-dir evaluation/results/m1_new_run
```

Python chạy lệnh thực nghiệm cần có Pillow, NumPy, OpenCV và Ultralytics như môi trường detection. Mỗi cấu hình ghi riêng thời gian ảnh đầu. Trung bình và p95 dùng các ảnh còn lại của cấu hình đó, tách theo nhóm. Ảnh đầu của B1 chạy sau B0 trong cùng tiến trình.

## Kiểm chứng

Toàn bộ 30 test qua khi chạy với OpenCV và FastAPI, gồm 18 test cũ. Venv gốc bỏ qua hai test tích hợp do thiếu thư viện. Coverage của `image_preprocessing.py` đạt 94%. Với một JPEG RGB không xoay và không resize, `best.pt` cho cùng hai detection, bbox, confidence và pixel crop như luồng decode cũ.

Docker Compose chuẩn dừng ở bước cài wheel CUDA do lỗi DNS đến `files.pythonhosted.org`. Image CPU kiểm thử dùng cùng build context, với PyTorch CPU được cài trước qua Dockerfile tạm. Trên `localhost:8001`, `/health` trả 200 và xác nhận model đã nạp, JPEG mẫu trả 200 và hai crop, bytes hỏng trả 400 ngay cả sau khi YOLO đã nạp model. M1 giữ tham chiếu đến hàm mở ảnh của Pillow trước khi Ultralytics thay hàm này bằng bản có cơ chế tự cài codec HEIF.

## Công cụ AI

Codex hỗ trợ viết mã, tạo fixture, gán nhãn ban đầu từ ảnh và soạn tài liệu. Các số liệu dựa trên lệnh chạy test và kết quả inference đã lưu.
