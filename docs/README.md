# Tài liệu dự án

Đọc [hướng dẫn chạy](SETUP.md), sau đó [bàn giao tích hợp cho M4](M4_HANDOFF.md). Đây là bộ tài liệu dùng chung theo code hiện tại; báo cáo merge và bản nháp theo từng phase đã được gom lại.

| Tài liệu | Khi cần đọc |
|---|---|
| [SETUP](SETUP.md) | Chuẩn bị model, dịch vụ, cache và khởi động |
| [M4_HANDOFF](M4_HANDOFF.md) | Tích hợp M1 → M2 → M3, việc M4 còn phải xác minh |
| [M2_CONTRACT](M2_CONTRACT.md) | Payload ROI/mask, hệ tọa độ, ID và fallback |
| [M3_HSV_PRODUCTION](M3_HSV_PRODUCTION.md) | HSV147, xây cache và chính sách khi thiếu màu |
| [EVALUATION](EVALUATION.md) | Kiểm thử, đánh giá M2/retrieval, cách hiểu kết quả |
| [DATA_SOURCES](DATA_SOURCES.md) | Nguồn ảnh, giấy phép và provenance nhãn |

Schema/config dùng chung nằm trong `m3_contract/`. Evidence máy đọc được giữ ở `m2/`, `m3_results/`, `m3_handoff_evidence/`, `m3_production_evidence/`; mở khi cần đối chiếu hoặc chạy lại. Mốc thời gian trong artifact là lịch sử lần chạy, không phải xác nhận deployment hiện tại.

Báo cáo cá nhân, slide, kế hoạch/progress M2 và tài liệu học nằm trong `docs/local/`, được Git bỏ qua. Clone mới không cần thư mục này để chạy ứng dụng hoặc test. `README.md` ở root là điểm vào; `AGENTS.md` là cấu hình coding agent.
