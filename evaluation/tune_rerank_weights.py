import argparse
import csv
import json
from pathlib import Path


NO_VALUE = None


DEFAULT_WEIGHT_GRID = [
    (0.70, 0.15, 0.15),
    (0.65, 0.15, 0.20),
    (0.60, 0.15, 0.25),
    (0.55, 0.20, 0.25),
    (0.50, 0.20, 0.30),
    (0.55, 0.15, 0.30),
    (0.50, 0.15, 0.35),
]


def normalize_terms(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip().lower() for value in values if str(value).strip()]


def product_text(product: dict) -> str:
    return " ".join(
        str(product.get(field) or "").lower()
        for field in ("id", "title", "category", "dominant_color_name")
    )


def normalize_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    image_weight, category_weight, color_weight = [max(0.0, float(value)) for value in weights]
    total = image_weight + category_weight + color_weight
    if total <= 0:
        return 0.65, 0.15, 0.20
    return image_weight / total, category_weight / total, color_weight / total


def score_product(product: dict, weights: tuple[float, float, float], use_penalties: bool) -> float:
    image_weight, category_weight, color_weight = normalize_weights(weights)
    score = (
        image_weight * float(product.get("image_similarity") or 0.0)
        + category_weight * float(product.get("category_score") or 0.0)
        + color_weight * float(product.get("color_score") or 0.0)
    )
    if use_penalties:
        score -= float(product.get("category_penalty") or 0.0)
        score -= float(product.get("color_penalty") or 0.0)
    return max(0.0, score)


def reranked_detected_items(detected_items: list[dict], weights: tuple[float, float, float], top_k: int, use_penalties: bool):
    reranked = []
    for item in detected_items:
        products = [dict(product) for product in item.get("top_products") or []]
        products.sort(key=lambda product: score_product(product, weights, use_penalties), reverse=True)
        next_item = dict(item)
        next_item["top_products"] = products[:top_k]
        reranked.append(next_item)
    return reranked


def all_top_products(detected_items: list[dict], top_k: int = 5) -> list[dict]:
    products = []
    for detected_index, item in enumerate(detected_items):
        for rank, product in enumerate((item.get("top_products") or [])[:top_k], start=1):
            products.append(
                {
                    "detected_index": detected_index,
                    "rank": rank,
                    "detected_class": item.get("class_name"),
                    "product": product,
                }
            )
    return products


def category_hit(expected_item: dict, detected_items: list[dict]) -> bool | None:
    terms = normalize_terms(expected_item.get("expected_categories"))
    if not terms:
        return NO_VALUE
    for entry in all_top_products(detected_items):
        if any(term in product_text(entry["product"]) for term in terms):
            return True
    return False


def color_hit(expected_item: dict, detected_items: list[dict]) -> bool | None:
    colors = normalize_terms(expected_item.get("expected_color"))
    if not colors:
        return NO_VALUE
    for entry in all_top_products(detected_items):
        product_color = str(entry["product"].get("dominant_color_name") or "").strip().lower()
        if product_color and product_color in colors:
            return True
    return False


def recall_hit(expected_item: dict, detected_items: list[dict]) -> bool | None:
    acceptable_ids = set(normalize_terms(expected_item.get("acceptable_product_ids")))
    if not acceptable_ids:
        return NO_VALUE
    for entry in all_top_products(detected_items):
        product_id = str(entry["product"].get("id") or "").strip().lower()
        if product_id in acceptable_ids:
            return True
    return False


def reciprocal_rank(expected_item: dict, detected_items: list[dict]) -> float | None:
    acceptable_ids = set(normalize_terms(expected_item.get("acceptable_product_ids")))
    if not acceptable_ids:
        return NO_VALUE

    ranks = []
    for entry in all_top_products(detected_items):
        product_id = str(entry["product"].get("id") or "").strip().lower()
        if product_id in acceptable_ids:
            ranks.append(entry["rank"])
    return 0.0 if not ranks else 1.0 / min(ranks)


def ratio(values: list[bool]) -> float | None:
    if not values:
        return NO_VALUE
    return sum(1 for value in values if value) / len(values)


def mean(values: list[float]) -> float | None:
    if not values:
        return NO_VALUE
    return sum(values) / len(values)


def evaluate(report: dict, labels: list[dict], weights: tuple[float, float, float], top_k: int, use_penalties: bool) -> dict:
    labels_by_id = {str(label.get("query_id")): label for label in labels}
    category_values = []
    color_values = []
    recall_values = []
    reciprocal_ranks = []

    for query_result in report.get("per_query_results") or []:
        label = labels_by_id.get(str(query_result.get("query_id"))) or {}
        expected_items = label.get("expected_items") or []
        detected_items = reranked_detected_items(
            query_result.get("detected_items") or [],
            weights=weights,
            top_k=top_k,
            use_penalties=use_penalties,
        )

        for expected_item in expected_items:
            category_value = category_hit(expected_item, detected_items)
            color_value = color_hit(expected_item, detected_items)
            recall_value = recall_hit(expected_item, detected_items)
            rank_value = reciprocal_rank(expected_item, detected_items)

            if category_value is not NO_VALUE:
                category_values.append(category_value)
            if color_value is not NO_VALUE:
                color_values.append(color_value)
            if recall_value is not NO_VALUE:
                recall_values.append(recall_value)
            if rank_value is not NO_VALUE:
                reciprocal_ranks.append(rank_value)

    metrics = {
        "category_at_5": ratio(category_values),
        "color_at_5": ratio(color_values),
        "recall_at_5": ratio(recall_values),
        "mrr": mean(reciprocal_ranks),
    }
    metrics["selection_score"] = sum(
        value for key, value in metrics.items() if key != "mrr" and isinstance(value, float)
    ) + (metrics["mrr"] if isinstance(metrics["mrr"], float) else 0.0)
    return metrics


def parse_weights(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("weights must be image,category,color")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Grid-search rerank weights on an existing eval result.")
    parser.add_argument("--result", default="evaluation/results/real_outfits_wikimedia_current.json")
    parser.add_argument("--labels", default="evaluation/labels.real.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weights", action="append", type=parse_weights)
    parser.add_argument("--no-penalties", action="store_true")
    parser.add_argument("--output", default="evaluation/results/rerank_weight_grid.csv")
    args = parser.parse_args()

    report = json.loads(Path(args.result).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))
    weight_grid = args.weights or DEFAULT_WEIGHT_GRID

    rows = []
    for weights in weight_grid:
        normalized = normalize_weights(weights)
        metrics = evaluate(
            report,
            labels,
            weights=normalized,
            top_k=args.top_k,
            use_penalties=not args.no_penalties,
        )
        rows.append(
            {
                "image_weight": normalized[0],
                "category_weight": normalized[1],
                "color_weight": normalized[2],
                **metrics,
            }
        )

    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print("Best weights")
    print("============")
    print(
        f"image/category/color = "
        f"{best['image_weight']:.2f}/{best['category_weight']:.2f}/{best['color_weight']:.2f}"
    )
    for key in ("category_at_5", "color_at_5", "recall_at_5", "mrr", "selection_score"):
        print(f"{key}: {best[key]}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
