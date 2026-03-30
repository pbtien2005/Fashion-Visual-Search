import os
import boto3
import httpx
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

def get_r2_client():
    """Khởi tạo kết nối tới Cloudflare R2 (tương thích S3 API)"""
    endpoint_url = os.getenv('R2_ENDPOINT_URL')
    access_key = os.getenv('R2_ACCESS_KEY_ID')
    secret_key = os.getenv('R2_SECRET_ACCESS_KEY')

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError("Lỗi: Thiếu thông tin cấu hình Cloudflare R2 trong file .env")

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto' # Bắt buộc là 'auto' đối với R2
    )

def upload_image_to_r2(r2_client, image_url: str, bucket_name: str, object_name: str) -> str:
    """
    Tải ảnh từ URL gốc (Amazon) và đẩy thẳng lên bucket R2.
    """
    try:
        # Sử dụng httpx thế hệ mới để tải ảnh cực nhanh vào bộ nhớ (RAM)
        with httpx.Client(timeout=15.0) as client:
            response = client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content # Dữ liệu ảnh dạng raw bytes

        # Đẩy trực tiếp cục bytes này lên R2
        r2_client.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=image_bytes,
            ContentType='image/jpeg' 
        )
        
        # Trả về định dạng đường dẫn s3 chuẩn để lưu vào PostgreSQL sau này
        return f"s3://{bucket_name}/{object_name}"
        
    except httpx.HTTPError as http_err:
        print(f"Lỗi mạng khi tải ảnh {image_url}: {http_err}")
    except ClientError as s3_err:
        print(f"Lỗi quyền/kết nối khi upload lên R2: {s3_err}")
    except Exception as e:
        print(f"Lỗi không xác định với ảnh {image_url}: {e}")
    
    return None

# Đoạn code test thử 1 ảnh
if __name__ == "__main__":
    # Đảm bảo bạn đã điền R2_BUCKET_NAME và các key khác vào .env
    test_bucket = os.getenv("R2_BUCKET_NAME", "fashion-raw-images")
    test_image_url = "https://m.media-amazon.com/images/I/61xO2Bu9oJL._AC_SY445_SX342_QL70_FMwebp_.jpg"
    
    print("Đang kết nối R2...")
    client = get_r2_client()
    
    print("Đang tải và upload ảnh test...")
    result_path = upload_image_to_r2(client, test_image_url, test_bucket, "images/test_upload.jpg")
    
    if result_path:
        print(f"Upload thành công! Đường dẫn lưu trong DB sẽ là: {result_path}")