import json
import torch
import httpx
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures
import os
import boto3
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

r2_client = boto3.client(
    's3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name='auto'
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_ID = "openai/clip-vit-base-patch32"

print(f"Đang tải mô hình CLIP lên thiết bị: {DEVICE}...")
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

def fetch_image_for_ai(image_url: str):
    try:
        if image_url.startswith("s3://"):
            parsed = urlparse(image_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            response = r2_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
            return Image.open(BytesIO(image_bytes)).convert("RGB")
            
        else:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(image_url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
                
    except Exception as e:
        print(f"Lỗi tải ảnh ({image_url}): {e}")
        return None

def extract_vector(image) -> list:
    """Đưa ảnh qua mạng nơ-ron CLIP và trả về Vector 512 chiều"""
    try:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
        return features.cpu().numpy()[0].tolist()
    except Exception as e:
        print(f"Lỗi khi trích xuất vector: {e}")
        return None

def process_single_item(item: dict) -> dict:
    image_url = item.get("r2_image_url") 
    if not image_url:
        return item
    
    img = fetch_image_for_ai(image_url)
    if img:
        vector = extract_vector(img)
        if vector:
            item["embedding"] = vector
            print(f"-> Tạo vector thành công cho ID: {item.get('id')}")
            
    return item

def run_ai_pipeline(input_file="data/fashion_metadata.json", output_file="data/fashion_with_vectors.json"):
    print("\n=== BẮT ĐẦU GIAI ĐOẠN 2: TRÍCH XUẤT VECTOR ===")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}. Hãy chạy ETL trước!")
        return

    print(f"Đã load {len(items)} sản phẩm. Bắt đầu chạy đa luồng (Multiprocessing)...")
    processed_items = []
    
    max_workers = 10 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_item, items)
        
        for result in results:
            if "embedding" in result:
                processed_items.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_items, f, ensure_ascii=False)
        
    print(f"Hoàn tất! Đã trích xuất thành công {len(processed_items)} vectors.")
    print(f"Dữ liệu sẵn sàng nạp vào Database (PostgreSQL) tại: {output_file}")

if __name__ == "__main__":
    run_ai_pipeline()