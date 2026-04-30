from AI.color_utils import name_color
from db.database import get_db_connection


def backfill_color_names():
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")

    rows = conn.execute(
        """
        SELECT id, dominant_color_rgb
        FROM fashion_items
        WHERE dominant_color_rgb IS NOT NULL;
        """
    ).fetchall()

    updated = 0
    changed = 0
    for product_id, rgb in rows:
        new_name = name_color(rgb)
        if not new_name:
            continue

        result = conn.execute(
            """
            UPDATE fashion_items
            SET dominant_color_name = %s
            WHERE id = %s
              AND dominant_color_name IS DISTINCT FROM %s;
            """,
            (new_name, product_id, new_name),
        )
        updated += 1
        changed += result.rowcount or 0

    conn.commit()
    conn.close()
    print(f"Processed {updated} color rows. Updated {changed} product color names.")


if __name__ == "__main__":
    backfill_color_names()
