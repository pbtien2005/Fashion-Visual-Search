import os
from apify_client import ApifyClient
from dotenv import load_dotenv

# Load tạm biến môi trường từ file .env (sau này có thể thay bằng pydantic-settings)
load_dotenv()

def fetch_amazon_products(search_keyword: str, max_items: int = 50) -> list:
    """
    Gọi Apify Actor (Amazon Product Scraper) để lấy dữ liệu sản phẩm.
    """
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        raise ValueError("Lỗi: Chưa cấu hình APIFY_API_TOKEN trong file .env")

    print(f"Khởi tạo kết nối tới Apify... Đang tìm kiếm: '{search_keyword}'")
    client = ApifyClient(apify_token)

    # ID của Actor Junglee/Amazon-crawler mà bạn vừa chốt
    actor_id = "BG3WDrGdteHgZgbPK"

    # Cấu hình payload đầu vào cho Actor
    run_input = {
        "categoryOrProductUrls": [{"url": f"https://www.amazon.com/s?k={search_keyword.replace(' ', '+')}"}],
        "maxItems": max_items,
        "proxyCountry": "US", # Ép dùng proxy Mỹ để tránh bị Amazon chặn hiển thị ảnh gốc
    }

    try:
        print("Đang chạy Actor trên cloud (Có thể mất 1-2 phút tùy số lượng)...")
        # Lệnh call() sẽ block luồng chạy và chờ cho đến khi Actor cào xong toàn bộ
        run = client.actor(actor_id).call(run_input=run_input)
        
        print("Chạy xong! Đang trích xuất dữ liệu từ Dataset...")
        # Lấy ID của cục data vừa cào được
        dataset_id = run["defaultDatasetId"]
        
        # Tải toàn bộ items về dạng list dictionary
        items = list(client.dataset(dataset_id).iterate_items())
        
        # Lọc ra những item hợp lệ (bắt buộc phải có ID và Link ảnh)
        valid_items = []
        for item in items:
            asin = item.get("asin")
            image_url = item.get("thumbnailImage")
            
            # Chỉ xử lý tiếp nếu sản phẩm có ID và Link ảnh hợp lệ
            if asin and image_url:
                
                # Xử lý trường giá (price) an toàn
                raw_price = item.get("price")
                price_value = raw_price.get("value") if isinstance(raw_price, dict) else None
                
                # Gom dữ liệu sạch thành một object mới
                clean_item = {
                    "asin": asin,
                    "title": item.get("title", "Chưa cập nhật tên"),
                    "brand": item.get("brand", "No Brand"),
                    "price": price_value,
                    "thumbnailImage": image_url,
                    "url": item.get("url"),
                    "category": item.get("breadCrumbs")
                }
                valid_items.append(clean_item)
                

        print(f"Thành công: Lấy được {len(valid_items)} sản phẩm hợp lệ.")
        return valid_items

    except Exception as e:
        print(f"Lỗi trong quá trình gọi Apify: {e}")
        return []

# Đoạn code test thử luồng chạy độc lập
if __name__ == "__main__":
    # Nhớ điền APIFY_API_TOKEN=... vào file .env ở cùng thư mục trước khi chạy
    sample_data = fetch_amazon_products(search_keyword="men graphic t-shirt", max_items=10)
    
    if sample_data:
        print("\n--- DỮ LIỆU MẪU (Sản phẩm đầu tiên) ---")
        first_item = sample_data[0]
        print(f"ID (ASIN): {first_item.get('asin')}")
        print(f"Tên: {first_item.get('title')[:50]}...") # Cắt ngắn tên cho đỡ rối
        print(f"Ảnh Thumbnail: {first_item.get('thumbnailImage')}")