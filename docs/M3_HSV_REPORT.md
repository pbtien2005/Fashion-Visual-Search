# M3: HSV Histogram — triển khai và kết quả

Audit trước sửa: [M3_AUDIT.md](M3_AUDIT.md). Checkout `main`, SHA `3247a5b33a016fd064ac9851410e7ec755d5e320`, working tree ban đầu sạch. Không reset, không ghi đè dữ liệu gốc.

## Code và phạm vi

- `AI/hsv_descriptor.py`: module độc lập OpenCV/NumPy, input RGB uint8 H×W×3; mask bool hoặc uint8 nhị phân 0/1/255, cùng H×W. Sai ảnh thì báo lỗi; sai/thiếu/rỗng mask theo policy `roi`, `empty` hoặc `error`. Ghi mask_status, fallback_reason, fraction, số pixel, nguồn ROI. Không suy đoán mask và không loại pixel trắng trong ROI.
- `AI/features.py`: `extract_features(garment_id, embedding_image, roi_rgb, mask, config, metadata)` trả dataclass M3Features gồm FashionCLIP vector 512, model/version, ColorDescriptor và metadata. Encoder nhận đúng ảnh riêng dành cho encoder; màu nhận ROI nguồn. `color_input_from_detection` ưu tiên source_roi_base64, sau đó bbox trên original_rgb, cuối cùng legacy image_base64; ghi rõ nguồn/fallback. `extract_detection_features` là entry point M2→M3 bổ sung, nhận payload hiện tại và original_rgb để tách nguồn màu khỏi crop encoder. Mask PNG dùng tọa độ ROI. Không thay đổi HTTP schema hiện hành.
- `AI/color_cache.py`: CLI cache màu riêng, không tạo lại embedding. Key gồm ID, model/version, ROI policy, descriptor version và hash nội dung ảnh/mask. Reader từ chối sai version/model/policy/config/key. Output dùng exclusive create, từ chối ghi đè file đã tồn tại. Thiếu ảnh là failure, thiếu mask fallback được đếm, không tạo vector giả.
- `AI/embedding_model.py`: chỉ bổ sung kiểm tra output NaN/Inf, norm L2 khác 1 và vector zero. Giữ nguyên model ID mặc định `patrickjohncyh/fashion-clip`, version 2, CLIPProcessor, đầu vào, no_grad và L2 normalization. Không tải/fine-tune model hoặc chạy lại catalog embedding.
- `evaluation/run_color_benchmark.py`: baseline RGB giữ nguyên; factorial H-S/H-S+achromatic × ROI/mask, thêm histogram intersection. Ghi ranking từng query, lỗi, coverage, extraction latency, hash input/source, environment, commit/status; paired cluster bootstrap 2.000 mẫu, seed 20261107.
- `tests/test_hsv_descriptor.py`: unit/integration tests cho descriptor, ROI adapter, cache, encoder output và benchmark failure denominator. `requirements.txt` thêm NumPy và opencv-python-headless. `.gitignore` cho phép lưu report/cache chẩn đoán riêng trong docs.
- `docs/m3_results/`: kết quả synthetic/photo và cache màu thực tế của 28 ảnh starter. Không thay đổi YOLO, DB, frontend, API response hoặc weights/penalties M4. Module HSV chưa được bật trong ranking production; RGB vẫn là baseline đang hoạt động.

## Công thức và cấu hình

OpenCV `COLOR_RGB2HSV`, H ∈ [0,180), S ∈ [0,256). Index H=`floor(H*18/180)`, S=`floor(S*8/256)`, flatten H-major. H-S baseline 18×8=144 chiều, L1 normalization. Đây là cấu hình khởi đầu, chưa chứng minh tối ưu.

Biến thể 147 chiều thêm Black/Gray/White. Trên thang uint8 OpenCV, mặc định low S là `S<=32`; Black nếu low S và `V<=40`; White nếu low S và `V>=220`; Gray nếu low S và `40<V<220`. Mọi pixel khác vào nhánh H-S. Pixel tối nhưng còn bão hòa vẫn ở nhánh có sắc. Mỗi pixel chỉ góp đúng một nhánh; toàn bộ 147 phần tử chia số pixel hợp lệ, tổng bằng 1. Không có pixel thì vector zero, được đánh dấu valid_pixels=0 và không được cache như descriptor hợp lệ.

Mask mặc định: min fraction .001, max fraction 1, min pixels 1; tất cả cấu hình được lưu và tham gia hash version `hsv-hs-v1-<dimension>-<hash>`. Những ngưỡng này chưa được chọn trên validation. Dùng file JSON `--config` để thay đổi, không chọn ngưỡng trên test set. Các policy khác nhau cũng tạo version khác nhau để ngăn trộn cache.

Với p,q đã L1-normalized: BC=`sum(sqrt(p_i*q_i))`, Hellinger distance H=`sqrt(max(0,1-BC))`, similarity=`1-H`. Đây là dạng khoảng cách Bhattacharyya/Hellinger của OpenCV, không dùng `-log(BC)`. Đối chứng intersection=`sum(min(p_i,q_i))`. Nếu một trong hai rỗng, similarity=0 kể cả rỗng/rỗng. Kiểm tra version, config, chiều, finite, không âm và tổng L1 trước so sánh. Similarity không phải xác suất đúng sản phẩm. Không chuyển ngưỡng RGB sang histogram.

## Kết quả đo đã chạy

| Bộ đo | Catalog/query | RGB ROI MRR | H-S ROI MRR | H-S+achromatic ROI MRR | H-S mask MRR | H-S+achromatic mask MRR |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic known foreground | 8/24 | .8368 | .5840 | .7569 | .6979 | .8750 |
| Ảnh starter, thêm nền có kiểm soát | 28/28 | .8405 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Synthetic ablation MRR: achromatic khi không mask +.1729, khi có mask +.1771; mask với H-S +.1139, với H-S+achromatic +.1181. Tuy nhiên H-S+achromatic không mask thấp hơn RGB -.0799. RGB có mask đạt MRR 1.0; H-S làm mất V nên xám/đen/trắng hoặc các màu khác độ sáng vẫn có thể không phân biệt được. H-S+achromatic vẫn không mô tả đủ độ sáng/mẫu vải.

Photo ablation: cả H-S và H-S+achromatic đều đạt ceiling, nên không phân biệt được lợi ích riêng của mask/achromatic. Query là chính ảnh catalog với một nền xanh được thêm vào; mask chỉ loại phần nền được thêm, giữ toàn bộ background vốn có trong ảnh. Đây là chẩn đoán độ ổn định với nền trên ảnh thật, **không phải** query người dùng độc lập, cũng không phải segmentation quần áo. Không suy ra cải thiện retrieval thực tế từ kết quả này hoặc bootstrap CI của nó.

Mỗi bộ đều 100% coverage ảnh/query, không lỗi. Cache catalog mới trích được 28/28, mask missing 28/28, fallback ROI 100%. Không có mask quần áo được gán nhãn, không có catalog vector gốc trong checkout để đánh giá pipeline FashionCLIP→pgvector→M4. Không khẳng định thuật toán mới cải thiện chất lượng tìm kiếm. Xem [synthetic report](m3_results/synthetic/report.md), [photo report](m3_results/photos/report.md); JSON cùng thư mục có per-query, CI, hashes và latency p50/p95. Latency chỉ đo trích màu, gồm overhead validation, chưa warm-up có kiểm soát, không phải latency API hay full retrieval.

## Tái lập và sử dụng

Trong môi trường đã cài `requirements.txt`, từ root repository:

```powershell
python -m unittest discover -s tests -v
python -m evaluation.run_color_benchmark --suite synthetic --output output/m3_synthetic_run2
python -m evaluation.run_color_benchmark --suite photos --output output/m3_photos_run2
python -m AI.color_cache --manifest docs/m3_results/catalog_manifest.json --output output/m3_catalog_run2.json
```

Output file/thư mục phải chưa tồn tại. Thư mục cha của cache output phải tồn tại. Môi trường local đã có NumPy/OpenCV/Torch/Transformers nhưng thiếu boto3 và pgvector; cài hai package vào `.venv/Lib/site-packages`, dùng `.venv` với system-site-packages cho test hồi quy. Không dùng model download hoặc service DB/YOLO trong unit tests. Test encoder dùng tensor output giả lập để kiểm tra đường normalize/validate thực tế; chưa chạy inference FashionCLIP với weights thật.

Ví dụ gọi độc lập:

```python
import numpy as np
from PIL import Image
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor, histogram_similarity

roi = np.asarray(Image.open("garment_roi.png").convert("RGB"))
config = HSVConfig(achromatic=True, mask_policy="roi")
descriptor = extract_hsv_descriptor(roi, config=config,
    metadata={"roi_source": "source_roi_rgb", "roi_policy": "source_roi_v1"})
score = histogram_similarity(descriptor, descriptor)
```

Không thể tự phát hiện thứ tự RGB/BGR từ các giá trị pixel. Nếu ảnh đến từ cv2.imread/imdecode, caller phải dùng `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` đúng một lần. PIL convert RGB dùng cho PNG/JPEG trong adapter/cache.

Đánh giá dữ liệu riêng bằng `--suite manifest --manifest <path> --output <new-dir>`; paths tương đối với manifest:

```json
{
  "catalog": [{"id": "p1", "image": "catalog/p1_roi.png", "mask": "catalog/p1_mask.png"}],
  "queries": [{"id": "q1", "image": "queries/q1_roi.png", "mask": "queries/q1_mask.png", "relevant_ids": ["p1"]}]
}
```

Ảnh phải là ROI nguồn đã cắt đúng, chưa gamma/padding/resize cho encoder. Manifest hỗ trợ missing/invalid mask theo policy, lỗi ảnh vẫn vào denominator. Relevant IDs phải tồn tại trong catalog manifest, kể cả ảnh catalog bị thiếu. Các metric dùng top-k thực được cấu hình, recall là số relevant được tìm / tổng relevant; MRR xét toàn bộ ranking màu. Tie-break ID tăng dần. Không lọc bỏ query khó hoặc failures. Chia validation/test theo garment ID và nguồn chụp để tránh leakage; validation chọn bins/threshold/policy trước, test giữ cố định. Benchmark này đánh giá màu riêng, không đại diện weighted reranking M4.

Phiên bản thực tế lúc kiểm thử: Python 3.12.10, NumPy 2.1.3, OpenCV 5.0.0, Pillow 11.3.0, Torch 2.7.1+cpu, Transformers 4.57.1, boto3 1.42.78, pgvector 0.4.2. Torch/Transformers trong môi trường sẵn có khác pin của repository (2.11.0/4.57.0); chưa xác minh lại trên đúng toàn bộ dependency pins hoặc với model weights thật. Các version encoder khai báo trong cache là metadata hợp đồng, không phải bằng chứng đã chạy inference trên catalog.

## Phối hợp tiếp theo

- M2: cung cấp lossless source_roi_base64 hoặc ảnh RGB nguồn+bbox; mask PNG binary cùng tọa độ/độ phân giải, metadata ROI policy, confidence và provenance. Không dùng crop 224 làm nguồn màu khi có source ROI hợp lệ. Adapter đã sẵn sàng; chưa thay schema M2 vì segmentation chưa có.
- M4: lấy `M3Features.color` và cache reader cùng version; thống nhất ROI policy query/catalog, kiểm tra coverage. Giữ nguyên encoder, candidate pool và weights khi thử thay color score; đo riêng calibration/penalty/gates histogram trên validation. Không tái sử dụng mismatch cap .35 hay palette-name penalty của RGB. Chỉ bật sau independent catalog/query benchmark có relevance judgments và masks thật, báo paired deltas/CI và failures đầy đủ.

## Kiểm chứng

Toàn bộ suite: **47 pass, 0 fail, 0 error**, gồm 18 test cũ và 29 test mới; unittest báo 0.483 giây thực thi test (không tính thời gian import Torch/Transformers). Lệnh đã chạy: `.venv\Scripts\python -m unittest discover -s tests -v`. Log: [tests.txt](m3_results/tests.txt). `git diff --check` pass. Hai lần chạy benchmark cho cùng ranking metrics và paired deltas; latency có dao động nên không dùng để kết luận hiệu năng tối ưu. Không có migration pgvector hoặc re-embedding toàn bộ catalog.
