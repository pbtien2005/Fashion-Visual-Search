import os
import json
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from AI.hsv_descriptor import ColorDescriptor
from AI.production_color import descriptor_display, production_config, ROI_POLICY

load_dotenv()

VECTOR_DIMENSION = 512
DEFAULT_EMBEDDING_MODEL = "patrickjohncyh/fashion-clip"
DEFAULT_EMBEDDING_VERSION = 2


def validate_vector_dimension(vector, expected_dimension: int = VECTOR_DIMENSION):
    if vector is None:
        raise ValueError("Vector is required")
    if len(vector) != expected_dimension:
        raise ValueError(f"Expected vector dimension {expected_dimension}, got {len(vector)}")


def get_configured_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_configured_embedding_version() -> int:
    raw_version = os.getenv("EMBEDDING_VERSION", str(DEFAULT_EMBEDDING_VERSION)).strip()
    try:
        return int(raw_version)
    except ValueError:
        return DEFAULT_EMBEDDING_VERSION

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
        print(f"Database connection failed: {e}")
        return None

def init_db():
    """Khởi tạo cấu trúc Database (Schema)"""
    conn = get_db_connection()
    if not conn: return
    
    print("Setting up database schema...")
    
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fashion_items (
            id VARCHAR(50) PRIMARY KEY,
            title TEXT,
            brand VARCHAR(100),
            price NUMERIC(10, 2),
            category TEXT,
            product_url TEXT,
            r2_image_url TEXT,
            dominant_color_rgb INTEGER[],
            dominant_color_name TEXT,
            embedding_model TEXT DEFAULT 'openai/clip-vit-base-patch32',
            embedding_version INTEGER DEFAULT 1,
            embedding VECTOR(512)
        );
    """)

    conn.execute("ALTER TABLE fashion_items ADD COLUMN IF NOT EXISTS product_url TEXT;")
    conn.execute("ALTER TABLE fashion_items ADD COLUMN IF NOT EXISTS dominant_color_rgb INTEGER[];")
    conn.execute("ALTER TABLE fashion_items ADD COLUMN IF NOT EXISTS dominant_color_name TEXT;")
    conn.execute("ALTER TABLE fashion_items ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'patrickjohncyh/fashion-clip';")
    conn.execute("ALTER TABLE fashion_items ADD COLUMN IF NOT EXISTS embedding_version INTEGER DEFAULT 2;")
    conn.execute("ALTER TABLE fashion_items ALTER COLUMN embedding_model SET DEFAULT 'patrickjohncyh/fashion-clip';")
    conn.execute("ALTER TABLE fashion_items ALTER COLUMN embedding_version SET DEFAULT 2;")
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS fashion_embedding_idx 
        ON fashion_items USING hnsw (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    conn.close()
    print("-> Database schema is ready.")

def load_data_to_db(input_file="data/fashion_with_hsv_vectors.json"):
    """Đọc file JSON và đẩy vào PostgreSQL"""
    if not os.path.exists(input_file):
        print(f"Missing input file: {input_file}.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    conn = get_db_connection()
    if not conn: return
    
    register_vector(conn)

    print(f"Loading {len(items)} products into database...")
    inserted_count = 0
    
    for item in items:
        item_id = item.get("id") or item.get("asin")
        vector = item.get("embedding")
        
        if not item_id or not vector:
            continue
        validate_vector_dimension(vector)
            
        price = item.get("price")
        # Preserve DB columns/API shape, but derive new display metadata only
        # from verified HSV. Legacy RGB values are not a descriptor fallback.
        dominant_color_rgb, dominant_color_name = None, None
        if item.get("color_descriptor"):
            color = ColorDescriptor(**item["color_descriptor"])
            if color.config != production_config().spec() or color.roi_policy != ROI_POLICY:
                raise ValueError("Catalog HSV config/ROI policy mismatch")
            rgb, dominant_color_name = descriptor_display(color)
            dominant_color_rgb = list(rgb) if rgb else None
        
        try:
            conn.execute("""
                INSERT INTO fashion_items (
                    id, title, brand, price, category, product_url, r2_image_url,
                    dominant_color_rgb, dominant_color_name, embedding_model,
                    embedding_version, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    brand = EXCLUDED.brand,
                    price = EXCLUDED.price,
                    category = EXCLUDED.category,
                    product_url = EXCLUDED.product_url,
                    r2_image_url = EXCLUDED.r2_image_url,
                    dominant_color_rgb = COALESCE(EXCLUDED.dominant_color_rgb, fashion_items.dominant_color_rgb),
                    dominant_color_name = COALESCE(EXCLUDED.dominant_color_name, fashion_items.dominant_color_name),
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_version = EXCLUDED.embedding_version,
                    embedding = EXCLUDED.embedding;
            """, (
                item_id, 
                item.get("title", ""), 
                item.get("brand", ""), 
                price, 
                str(item.get("category", "")), 
                item.get("original_url") or item.get("product_url"),
                item.get("r2_image_url") or item.get("thumbnailImage"), 
                dominant_color_rgb,
                dominant_color_name,
                item.get("embedding_model") or get_configured_embedding_model(),
                item.get("embedding_version") or get_configured_embedding_version(),
                vector
            ))
            inserted_count += 1
        except Exception as e:
            print(f"Insert failed for ID {item_id}: {e}")
            conn.rollback() 
            continue
            
    conn.commit()
    conn.close()
    
    print(f"Done. Loaded {inserted_count} records into PostgreSQL.")

def _rows_to_products(rows):
    return [
        {
            "id": row[0],
            "title": row[1],
            "brand": row[2],
            "price": float(row[3]) if row[3] is not None else None,
            "category": row[4],
            "product_url": row[5],
            "image_url": row[6],
            "similarity": float(row[7]) if row[7] is not None else None,
            "dominant_color_rgb": list(row[8]) if row[8] is not None else None,
            "dominant_color_name": row[9],
            "embedding_model": row[10],
            "embedding_version": int(row[11]) if row[11] is not None else None,
        }
        for row in rows
    ]


def search_similar_by_vector(
    vector,
    top_k: int = 5,
    category_terms=None,
    embedding_model: str | None = None,
    embedding_version: int | None = None,
):
    """Search pgvector by CLIP vector.

    Similarity uses 1 - cosine_distance because pgvector `<=>` returns cosine distance.
    `category_terms` remains for backward compatibility, but current retrieval keeps
    candidates broad and does not hard-filter by category.
    """
    validate_vector_dimension(vector)
    top_k = max(1, min(int(top_k), 100))
    embedding_model = embedding_model or get_configured_embedding_model()
    embedding_version = embedding_version or get_configured_embedding_version()

    conn = get_db_connection()
    if not conn:
        return []

    register_vector(conn)
    try:
        query = f"""
            SELECT id, title, brand, price, category, product_url, r2_image_url,
                   1 - (embedding <=> %s::vector) AS similarity,
                   dominant_color_rgb, dominant_color_name,
                   embedding_model, embedding_version
            FROM fashion_items
            WHERE embedding_model = %s
              AND embedding_version = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        rows = conn.execute(query, (vector, embedding_model, embedding_version, vector, top_k)).fetchall()
        return _rows_to_products(rows)
    finally:
        conn.close()


def list_products(limit: int = 50, offset: int = 0, q: str | None = None, category: str | None = None):
    """List products from the catalog with optional text/category filters."""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    q = (q or "").strip()
    category = (category or "").strip()

    conn = get_db_connection()
    if not conn:
        return {"items": [], "total": 0}

    try:
        where_parts = []
        params = []

        if q:
            pattern = f"%{q}%"
            where_parts.append("(title ILIKE %s OR brand ILIKE %s OR category ILIKE %s)")
            params.extend([pattern, pattern, pattern])

        if category:
            where_parts.append("category ILIKE %s")
            params.append(f"%{category}%")

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        count_query = f"SELECT COUNT(*) FROM fashion_items {where_clause};"
        total = conn.execute(count_query, tuple(params)).fetchone()[0]

        data_query = f"""
            SELECT id, title, brand, price, category, product_url, r2_image_url,
                   NULL::double precision AS similarity,
                   dominant_color_rgb, dominant_color_name,
                   embedding_model, embedding_version
            FROM fashion_items
            {where_clause}
            ORDER BY title NULLS LAST, id
            LIMIT %s OFFSET %s;
        """
        rows = conn.execute(data_query, tuple(params + [limit, offset])).fetchall()
        return {"items": _rows_to_products(rows), "total": int(total)}
    finally:
        conn.close()


def get_product_by_id(product_id: str):
    conn = get_db_connection()
    if not conn:
        return None

    try:
        row = conn.execute(
            """
            SELECT id, title, brand, price, category, product_url, r2_image_url,
                   NULL::double precision AS similarity,
                   dominant_color_rgb, dominant_color_name,
                   embedding_model, embedding_version
            FROM fashion_items
            WHERE id = %s;
            """,
            (product_id,),
        ).fetchone()
        if not row:
            return None
        return _rows_to_products([row])[0]
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    load_data_to_db()
