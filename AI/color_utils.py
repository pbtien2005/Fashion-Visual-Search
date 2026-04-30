import colorsys
import math
from collections import defaultdict

from PIL import Image


PALETTE = {
    "black": (20, 20, 20),
    "navy": (24, 38, 74),
    "white": (240, 240, 240),
    "cream": (235, 228, 205),
    "gray": (128, 128, 128),
    "red": (180, 40, 40),
    "pink": (220, 120, 160),
    "orange": (220, 120, 40),
    "yellow": (220, 190, 60),
    "green": (60, 140, 80),
    "olive": (120, 118, 78),
    "blue": (50, 100, 190),
    "purple": (120, 70, 160),
    "brown": (120, 80, 45),
    "tan": (210, 180, 140),
    "khaki": (190, 169, 117),
    "beige": (205, 180, 135),
}

MAX_RGB_DISTANCE = math.sqrt(3 * (255**2))

COLOR_MISMATCH_CAPS = {
    frozenset(("black", "gray")): 0.65,
    frozenset(("black", "brown")): 0.45,
    frozenset(("black", "blue")): 0.45,
    frozenset(("black", "green")): 0.45,
    frozenset(("black", "navy")): 0.70,
    frozenset(("black", "khaki")): 0.25,
    frozenset(("black", "tan")): 0.25,
    frozenset(("black", "beige")): 0.25,
    frozenset(("black", "cream")): 0.20,
    frozenset(("white", "beige")): 0.65,
    frozenset(("white", "cream")): 0.82,
    frozenset(("white", "khaki")): 0.45,
    frozenset(("white", "tan")): 0.45,
    frozenset(("white", "gray")): 0.55,
    frozenset(("gray", "beige")): 0.55,
    frozenset(("gray", "khaki")): 0.45,
    frozenset(("gray", "tan")): 0.45,
    frozenset(("gray", "olive")): 0.45,
    frozenset(("gray", "brown")): 0.45,
    frozenset(("gray", "green")): 0.45,
    frozenset(("gray", "blue")): 0.45,
    frozenset(("navy", "blue")): 0.65,
    frozenset(("brown", "beige")): 0.65,
    frozenset(("brown", "tan")): 0.70,
    frozenset(("brown", "khaki")): 0.55,
    frozenset(("tan", "khaki")): 0.82,
    frozenset(("tan", "beige")): 0.78,
    frozenset(("tan", "cream")): 0.58,
    frozenset(("khaki", "beige")): 0.78,
    frozenset(("khaki", "cream")): 0.58,
    frozenset(("olive", "khaki")): 0.62,
    frozenset(("olive", "tan")): 0.50,
    frozenset(("brown", "orange")): 0.55,
    frozenset(("brown", "green")): 0.40,
    frozenset(("green", "blue")): 0.50,
    frozenset(("green", "olive")): 0.70,
    frozenset(("red", "pink")): 0.65,
    frozenset(("red", "orange")): 0.55,
    frozenset(("purple", "pink")): 0.55,
}

DEFAULT_COLOR_MISMATCH_CAP = 0.35


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


def _add_rgb_to_bucket(buckets, key, rgb):
    buckets[key][0] += rgb[0]
    buckets[key][1] += rgb[1]
    buckets[key][2] += rgb[2]
    buckets[key][3] += 1


def _color_from_buckets(buckets) -> tuple[int, int, int] | None:
    if not buckets:
        return None

    total_r, total_g, total_b, count = max(buckets.values(), key=lambda item: item[3])
    if count <= 0:
        return None

    return (
        round(total_r / count),
        round(total_g / count),
        round(total_b / count),
    )


def extract_dominant_color(
    image,
    max_size: int = 64,
    bucket_size: int = 24,
    center_ratio: float = 0.50,
    min_center_pixels: int = 32,
) -> tuple[int, int, int] | None:
    """Return a dominant RGB color with center crop priority.

    Clothing crops often contain background/person pixels near bbox borders.
    The center region is used first when it has enough non-background pixels;
    the whole crop remains a fallback for off-center or tiny garments.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))

    width, height = image.size
    center_ratio = max(0.10, min(1.0, float(center_ratio)))
    center_width = max(1, int(width * center_ratio))
    center_height = max(1, int(height * center_ratio))
    center_left = (width - center_width) // 2
    center_top = (height - center_height) // 2
    center_right = center_left + center_width
    center_bottom = center_top + center_height

    center_buckets = defaultdict(lambda: [0, 0, 0, 0])
    buckets = defaultdict(lambda: [0, 0, 0, 0])
    fallback_buckets = defaultdict(lambda: [0, 0, 0, 0])

    for y in range(height):
        for x in range(width):
            rgb = image.getpixel((x, y))
            key = tuple(channel // bucket_size for channel in rgb)
            _add_rgb_to_bucket(fallback_buckets, key, rgb)

            if _is_background_like(rgb):
                continue

            _add_rgb_to_bucket(buckets, key, rgb)
            if center_left <= x < center_right and center_top <= y < center_bottom:
                _add_rgb_to_bucket(center_buckets, key, rgb)

    center_color = _color_from_buckets(center_buckets)
    center_pixel_count = sum(bucket[3] for bucket in center_buckets.values())
    if center_color is not None and center_pixel_count >= min_center_pixels:
        return center_color

    selected_buckets = buckets or fallback_buckets
    return _color_from_buckets(selected_buckets)


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

    if query_name and product_name and query_name != product_name:
        cap = COLOR_MISMATCH_CAPS.get(
            frozenset((query_name, product_name)),
            DEFAULT_COLOR_MISMATCH_CAP,
        )
        return min(distance_score, cap)

    return distance_score
