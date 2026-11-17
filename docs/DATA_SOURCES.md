# Nguồn dữ liệu và nhãn

Giữ tác giả/trang nguồn/license và điều kiện attribution khi chia sẻ ảnh. Đây là metadata lưu lúc chuẩn bị dataset, không phải kiểm lại giấy phép website hiện tại.

| Bộ | Metadata nguồn |
|---|---|
| Real outfits Wikimedia | [danh sách](data_sources/real_outfits.md), [JSON](../evaluation/real_outfit_sources.json) |
| M1 validation/test | [validation CSV](../evaluation/m1_validation.csv), [test CSV](../evaluation/m1_test.csv) |
| M2 review2 | [manifest](../evaluation/m2/annotations.review2.json), [mask](../evaluation/m2/reference_masks/) |
| M2 regression | [manifest](../evaluation/m2/regression_manifest.json) |
| M3 catalog mẫu | [manifest](m3_results/catalog_manifest.json) |

M1 bổ sung 5 ảnh trong `evaluation/queries/m1/`: JPEG chuẩn hóa hướng, cạnh dài tối đa 1600 px, quality 93; `ostau.jpg` từ thumbnail Wikimedia 500 px. Giữ phơi sáng trong bước chuẩn bị. Overlay B0/B1 giữ license tương ứng; B1 có thể gamma theo config khóa.

M2 dùng 20 target chung từ M1 (15 ảnh dev + 5 ảnh test bổ sung), không phải 20 ảnh mới đóng góp riêng. Người dùng kiểm bbox/class/màu và 5 mask AI gợi ý; manifest ghi provenance/checksum/split/hệ tọa độ. Không mô tả mask là GT vẽ độc lập hoặc đánh giá chéo thành viên.

Query catalog chỉ là smoke/regression, dễ hơn outfit thực. Mask synthetic trong `tests/fixtures/m2/` chỉ kiểm hợp đồng/hình học; không trộn thành benchmark độc lập không nêu giới hạn.
