import colorsys
import math
from collections import defaultdict

from PIL import Image


PALETTE = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "red": (180, 40, 40),
    "pink": (220, 120, 160),
    "orange": (220, 120, 40),
    "yellow": (220, 190, 60),
    "green": (60, 140, 80),
    "blue": (50, 100, 190),
    "purple": (120, 70, 160),
    "brown": (120, 80, 45),
    "beige": (205, 180, 135),
}

MAX_RGB_DISTANCE = math.sqrt(3 * (255**2))


def ensure_rgb_tuple(color) -> tuple[int, int, int] | None:
    if not color or len(color) != 3:
        return None

    try:
        return tuple(max(0, min(255, int(channel))) for channel in color)
    except (TypeError, ValueError):
        return None


def _is_background_like(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return v > 0.88 and s < 0.18


def extract_dominant_color(image, max_size: int = 64, bucket_size: int = 24) -> tuple[int, int, int] | None:
    """Return a simple dominant RGB color, ignoring bright low-saturation backgrounds."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))

    buckets = defaultdict(lambda: [0, 0, 0, 0])
    fallback_buckets = defaultdict(lambda: [0, 0, 0, 0])

    for rgb in image.getdata():
        key = tuple(channel // bucket_size for channel in rgb)
        fallback_buckets[key][0] += rgb[0]
        fallback_buckets[key][1] += rgb[1]
        fallback_buckets[key][2] += rgb[2]
        fallback_buckets[key][3] += 1

        if _is_background_like(rgb):
            continue

        buckets[key][0] += rgb[0]
        buckets[key][1] += rgb[1]
        buckets[key][2] += rgb[2]
        buckets[key][3] += 1

    selected_buckets = buckets or fallback_buckets
    if not selected_buckets:
        return None

    total_r, total_g, total_b, count = max(selected_buckets.values(), key=lambda item: item[3])
    if count <= 0:
        return None

    return (
        round(total_r / count),
        round(total_g / count),
        round(total_b / count),
    )


def name_color(rgb) -> str | None:
    rgb = ensure_rgb_tuple(rgb)
    if rgb is None:
        return None

    return min(
        PALETTE,
        key=lambda name: math.dist(rgb, PALETTE[name]),
    )


def color_similarity(query_rgb, product_rgb, query_name: str | None = None, product_name: str | None = None) -> float:
    query_rgb = ensure_rgb_tuple(query_rgb)
    product_rgb = ensure_rgb_tuple(product_rgb)
    if query_rgb is None or product_rgb is None:
        return 0.0

    distance_score = 1.0 - min(math.dist(query_rgb, product_rgb) / MAX_RGB_DISTANCE, 1.0)
    if query_name and product_name and query_name == product_name:
        return max(distance_score, 0.85)
    return distance_score
