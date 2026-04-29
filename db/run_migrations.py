from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db.database import get_db_connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def run_migrations():
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        applied = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_migrations;").fetchall()
        }

        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = migration_path.name
            if version in applied:
                print(f"Skip migration {version} (already applied).")
                continue

            print(f"Apply migration {version}...")
            sql = migration_path.read_text(encoding="utf-8")
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s);", (version,))
            conn.commit()

        print("Migrations complete.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migrations()
