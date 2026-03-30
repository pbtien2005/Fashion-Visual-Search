import psycopg
from db.database import get_db_connection
from AI.vectorizer import fetch_image_for_ai, extract_vector

def search_similar_fashion(image_url: str, top_k: int = 5):
    print(f"1. Đang tải và phân tích ảnh đầu vào...")
    # Tái sử dụng hàm tải ảnh từ internet
    img = fetch_image_for_ai(image_url)
    if not img:
        print("Không thể tải ảnh. Vui lòng kiểm tra lại URL.")
        return
    
    # Đưa ảnh qua mô hình CLIP để lấy vector 512 chiều
    print(f"2. Đang trích xuất Vector bằng AI...")
    query_vector = extract_vector(img)
    if not query_vector:
        return

    print(f"3. Đang truy vấn Database (PostgreSQL + pgvector)...")
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        query = """
            SELECT id, title, brand, price, r2_image_url, 
                   1 - (embedding <=> %s::vector) AS similarity
            FROM fashion_items
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        cur.execute(query, (query_vector, query_vector, top_k))
        results = cur.fetchall()
        
        print("\n=== KẾT QUẢ TÌM KIẾM TỪ KHO HÀNG ===")
        for i, row in enumerate(results):
            similarity_score = row[5]
            print(f"Top {i+1} (Độ giống: {similarity_score * 100:.2f}%):")
            print(f"- ID:     {row[0]}")
            print(f"- Tên:    {row[1][:60]}...")
            print(f"- Giá:    ${row[3]} | Hãng: {row[2]}")
            print(f"- R2 URL: {row[4]}\n")
            
    except Exception as e:
        print(f"Lỗi truy vấn Database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Đưa vào một bức ảnh áo sơ mi nam kẻ sọc bất kỳ lấy từ trên mạng
    test_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVUuFRc992mmYkLPowCctNKMQGDCzGES6GTQ&s"
    
    print(f"=== TÌM KIẾM SẢN PHẨM TƯƠNG TỰ ===")
    print(f"Ảnh đầu vào: {test_image_url}\n")
    
    # Tìm 3 sản phẩm giống nhất
    search_similar_fashion(image_url=test_image_url, top_k=3)