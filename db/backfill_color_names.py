"""Optional display-field refresh from the official HSV sidecar, never RGB inference."""
from AI.production_color import descriptor_display, product_color
from db.database import (get_db_connection, get_configured_embedding_model,
                         get_configured_embedding_version)


def backfill_color_names():
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")

    rows = conn.execute(
        """
        SELECT id FROM fashion_items;
        """
    ).fetchall()

    updated = 0
    changed = 0
    for (product_id,) in rows:
        color = product_color(product_id, get_configured_embedding_model(), get_configured_embedding_version())
        rgb, new_name = descriptor_display(color)
        if not new_name:
            continue

        result = conn.execute(
            """
            UPDATE fashion_items
            SET dominant_color_name = %s, dominant_color_rgb = %s
            WHERE id = %s
              AND (dominant_color_name IS DISTINCT FROM %s OR dominant_color_rgb IS DISTINCT FROM %s);
            """,
            (new_name, list(rgb), product_id, new_name, list(rgb)),
        )
        updated += 1
        changed += result.rowcount or 0

    conn.commit()
    conn.close()
    print(f"Processed {updated} color rows. Updated {changed} product color names.")


if __name__ == "__main__":
    backfill_color_names()
