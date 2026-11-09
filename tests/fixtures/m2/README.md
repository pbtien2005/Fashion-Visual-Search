# Fixture hợp đồng M2

Sinh lại từ root bằng `python -m evaluation.prepare_m2_phase1` trong môi trường detection.

- `source_rgb.png`: ảnh tổng hợp 96×128; nền RGB (15,25,35), áo chữ T RGB (30,100,200), ô đỏ (240,20,10) ở góc trên trái, ô xanh lá ở góc dưới phải.
- `detection_rgb.png`: cộng bão hòa 40 đơn vị từng kênh, cùng hình học. Đây chỉ là fixture phân biệt hai nhánh, không phải thuật toán gamma M1.
- `source_roi.png`: lát cắt `[16:80,32:96]`, 64×64, chưa padding/resize.
- `reference_mask.png`: mask áo chữ T trong ROI, uint8 0/255; foreground=255. Chỉ có ground truth tổng hợp, không đại diện độ chính xác trên ảnh thật.
- `legacy_response.json`: response thật của serializer hiện tại với detector giả lập một bbox; ba crop JPEG 224×224.
- `candidate_response_unmasked.json` và `candidate_response_masked.json`: ví dụ hợp đồng đề xuất ở phase 1. Phase 2 đã triển khai trường ROI nguồn/unmasked; masked example vẫn chỉ dùng mask reference tổng hợp, không phải dự đoán Otsu.
- `phase2_response.json`: output thực của detector + serializer phase 2 với bbox giả lập trên hai nhánh RGB fixture; có ID, ROI nguồn PNG, metadata và mask null. Đây là fixture bàn giao M1/M3; không cần model để sinh bằng `python -m evaluation.check_m2_phase2 fixture`.
- `no_detection_response.json`: giữ hành vi cũ của detection service (`success=false`, crop rỗng); API chính có thể chuyển thành kết quả tìm kiếm rỗng thành công.
- `fixture_manifest.json`: bbox hợp lệ, chạm biên, ngoài ảnh, nhỏ và sai; x2/y2 exclusive. Các bbox này dành cho kiểm thử phase 2, chưa xác nhận code hiện tại đã clip đúng.
- `contact_sheet.png`: xem nhanh nguồn, nhánh sáng, ROI và mask.

Fixture không cần YOLO weights, DB, FashionCLIP hoặc mạng. Cần NumPy/OpenCV/Pillow/Pydantic trong image detection. Consumer schema cũ bỏ qua trường mới; test này không thay thế decoder M3 sẽ làm ở phase 5. Đã kiểm tra schema cũ từ image phase 1 đọc được response HTTP phase 2.
