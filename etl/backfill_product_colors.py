"""Compatibility CLI for HSV147 sidecar rebuilds; requires --catalog and --output.

Does not write the old RGB catalog/DB cache. See AI.rebuild_color_cache.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AI.rebuild_color_cache import main


def backfill_product_colors():
    return main()


if __name__ == "__main__":
    backfill_product_colors()
