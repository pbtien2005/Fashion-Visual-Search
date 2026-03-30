import os
import json
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    try:
        conn = psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5433"),
            user=os.getenv("DB_USER", "pbtien"),
            password=os.getenv("DB_PASSWORD", "tien12345"),
            dbname=os.getenv("DB_NAME", "fashion_db")
        )
        return conn
    except Exception as e:
        print(f"Lỗi kết nối Database: {e}")
        return None

def init_db():
    """Khởi tạo cấu trúc Database (Schema)"""
    conn = get_db_connection()
    if not conn: return
    
    print("Đang thiết lập Database Schema...")
    
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fashion_items (
            id VARCHAR(50) PRIMARY KEY,
            title TEXT,
            brand VARCHAR(100),
            price NUMERIC(10, 2),
            category TEXT,
            r2_image_url TEXT,
            embedding VECTOR(512)
        );
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS fashion_embedding_idx 
        ON fashion_items USING hnsw (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    conn.close()
    print("-> Thiết lập Database thành công!")

def load_data_to_db(input_file="data/fashion_with_vectors.json"):
    """Đọc file JSON và đẩy vào PostgreSQL"""
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    conn = get_db_connection()
    if not conn: return
    
    register_vector(conn)

    print(f"Bắt đầu nạp {len(items)} sản phẩm vào Database...")
    inserted_count = 0
    
    for item in items:
        item_id = item.get("id") or item.get("asin")
        vector = item.get("embedding")
        
        if not item_id or not vector:
            continue
            
        price = item.get("price")
        
        try:
            conn.execute("""
                INSERT INTO fashion_items (id, title, brand, price, category, r2_image_url, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    price = EXCLUDED.price,
                    r2_image_url = EXCLUDED.r2_image_url,
                    embedding = EXCLUDED.embedding;
            """, (
                item_id, 
                item.get("title", ""), 
                item.get("brand", ""), 
                price, 
                str(item.get("category", "")), 
                item.get("r2_image_url") or item.get("thumbnailImage"), 
                vector
            ))
            inserted_count += 1
        except Exception as e:
            print(f"Lỗi khi insert ID {item_id}: {e}")
            conn.rollback() 
            continue
            
    conn.commit()
    conn.close()
    
    print(f"Hoàn tất! Đã nạp thành công {inserted_count} record vào PostgreSQL.")

if __name__ == "__main__":
    init_db()
    load_data_to_db()