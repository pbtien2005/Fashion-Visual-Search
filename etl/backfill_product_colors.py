import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AI.color_utils import extract_dominant_color, name_color
from AI.vectorizer import fetch_image_for_ai
from db.database import get_db_connection

load_dotenv()


def _fetch_pending_products(conn, batch_size: int):
    return conn.execute(
        """
        SELECT id, r2_image_url
        FROM fashion_items
        WHERE dominant_color_rgb IS NULL
          AND r2_image_url IS NOT NULL
        ORDER BY id
        LIMIT %s;
        """,
        (batch_size,),
    ).fetchall()


def backfill_product_colors():
    batch_size = max(1, int(os.getenv("COLOR_BACKFILL_BATCH_SIZE", "50")))
    max_items = int(os.getenv("COLOR_BACKFILL_LIMIT", "0"))
    processed = 0
    updated = 0
    skipped = 0

    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")

    try:
        while True:
            if max_items and processed >= max_items:
                break

            rows = _fetch_pending_products(conn, batch_size)
            if not rows:
                break

            for item_id, image_url in rows:
                if max_items and processed >= max_items:
                    break

                processed += 1
                image = fetch_image_for_ai(image_url)
                if image is None:
                    print(f"Skip {item_id}: could not fetch image")
                    skipped += 1
                    continue

                rgb = extract_dominant_color(image)
                color_name = name_color(rgb)
                if rgb is None or color_name is None:
                    print(f"Skip {item_id}: could not extract color")
                    skipped += 1
                    continue

                conn.execute(
                    """
                    UPDATE fashion_items
                    SET dominant_color_rgb = %s,
                        dominant_color_name = %s
                    WHERE id = %s;
                    """,
                    (list(rgb), color_name, item_id),
                )
                updated += 1
                print(f"Updated {item_id}: {color_name} {rgb}")

            conn.commit()

        print(f"Color backfill complete. processed={processed}, updated={updated}, skipped={skipped}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    backfill_product_colors()
