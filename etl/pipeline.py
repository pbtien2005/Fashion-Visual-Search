import os
import json
from dotenv import load_dotenv

from etl.scraper import fetch_amazon_products
from etl.r2_uploader import get_r2_client, upload_image_to_r2

load_dotenv()

def run_etl_pipeline(search_keyword: str, max_items: int = 50, output_file: str = "data/fashion_metadata.json"):
    print(f"=== BẮT ĐẦU PIPELINE ETL CHO TỪ KHÓA: '{search_keyword}' ===")
    
    # 1. KHỞI TẠO BỘ NHỚ (Đọc lịch sử cào)
    history_file = "data/scraped_history.json"
    seen_ids = set()
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            seen_ids = set(json.load(f))
        print(f"Đã tải {len(seen_ids)} ID từ lịch sử để lọc trùng.")

    # 2. EXTRACT
    items = fetch_amazon_products(search_keyword=search_keyword, max_items=max_items)
    if not items:
        return

    r2_client = get_r2_client()
    bucket_name = os.getenv("R2_BUCKET_NAME")
    processed_data = []

    # 3. TRANSFORM & LOAD với bộ lọc
    for index, item in enumerate(items):
        item_id = item.get("asin")
        
        # KIỂM TRA TRÙNG LẶP Ở ĐÂY
        if item_id in seen_ids:
            print(f"[{index + 1}/{len(items)}] Bỏ qua {item_id} (Đã có trong hệ thống).")
            continue
            
        image_url = item.get("thumbnailImage")
        print(f"[{index + 1}/{len(items)}] Đang xử lý sản phẩm MỚI: {item_id}")
        
        object_name = f"images/{item_id}.jpg"
        r2_path = upload_image_to_r2(r2_client, image_url, bucket_name, object_name)
        
        if r2_path:
            processed_item = {
                "id": item_id,
                "title": item.get("title"),
                "brand": item.get("brand"),
                "price": item.get("price"),
                "category": item.get("category"),
                "original_url": item.get("url"),
                "r2_image_url": r2_path
            }
            processed_data.append(processed_item)
            
            # Đánh dấu là đã xử lý xong để đưa vào lịch sử
            seen_ids.add(item_id)

    # 4. LƯU TRỮ METADATA VÀ CẬP NHẬT LỊCH SỬ
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lưu metadata của đợt chạy này
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
    # Lưu lại bộ nhớ (danh sách ID) cho đợt chạy ngày mai
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(list(seen_ids), f)
        
    print(f"Hoàn tất! Cập nhật thêm {len(processed_data)} sản phẩm mới.")

if __name__ == "__main__":
    run_etl_pipeline(search_keyword="men graphic t-shirt", max_items=15)