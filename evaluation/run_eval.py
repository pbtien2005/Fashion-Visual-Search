import argparse
import csv
import json
import mimetypes
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import httpx


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
NO_VALUE = None


def _normalize_terms(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _product_text(product: dict) -> str:
    return " ".join(
        str(product.get(field) or "").lower()
        for field in ("id", "title", "brand", "category", "dominant_color_name")
    )


def _all_top_products(detected_items: list[dict], top_k: int = 5) -> list[dict]:
    products = []
    for detected_index, item in enumerate(detected_items):
        item_products = item.get("similar_products") or item.get("low_confidence_products") or []
        for rank, product in enumerate(item_products[:top_k], start=1):
            products.append(
                {
                    "detected_index": detected_index,
                    "rank": rank,
                    "detected_class": item.get("class_name"),
                    "product": product,
                }
            )
    return products


def category_hit_at_5(expected_item: dict, detected_items: list[dict]) -> bool | None:
    terms = _normalize_terms(expected_item.get("expected_categories"))
    if not terms:
        return NO_VALUE

    for entry in _all_top_products(detected_items):
        text = _product_text(entry["product"])
        if any(term in text for term in terms):
            return True
    return False


def color_hit_at_5(expected_item: dict, detected_items: list[dict]) -> bool | None:
    colors = _normalize_terms(expected_item.get("expected_color"))
    if not colors:
        return NO_VALUE

    for entry in _all_top_products(detected_items):
        product_color = str(entry["product"].get("dominant_color_name") or "").strip().lower()
        if product_color and product_color in colors:
            return True
    return False


def recall_hit_at_5(expected_item: dict, detected_items: list[dict]) -> bool | None:
    acceptable_ids = set(_normalize_terms(expected_item.get("acceptable_product_ids")))
    if not acceptable_ids:
        return NO_VALUE

    for entry in _all_top_products(detected_items):
        product_id = str(entry["product"].get("id") or "").strip().lower()
        if product_id in acceptable_ids:
            return True
    return False


def reciprocal_rank(expected_item: dict, detected_items: list[dict]) -> float | None:
    acceptable_ids = set(_normalize_terms(expected_item.get("acceptable_product_ids")))
    if not acceptable_ids:
        return NO_VALUE

    best_rank = None
    for entry in _all_top_products(detected_items):
        product_id = str(entry["product"].get("id") or "").strip().lower()
        if product_id in acceptable_ids:
            rank = int(entry["rank"])
            best_rank = rank if best_rank is None else min(best_rank, rank)

    return 0.0 if best_rank is None else 1.0 / best_rank


def summarize_detection_item(item: dict) -> dict:
    products = item.get("similar_products") or item.get("low_confidence_products") or []
    return {
        "class_name": item.get("class_name"),
        "class_name_vi": item.get("class_name_vi"),
        "confidence": item.get("confidence"),
        "bbox": item.get("bbox"),
        "is_confident_match": item.get("is_confident_match"),
        "message": item.get("message"),
        "query_color_name": item.get("query_color_name"),
        "query_color_rgb": item.get("query_color_rgb"),
        "similar_product_count": len(products),
        "low_confidence_product_count": len(item.get("low_confidence_products") or []),
        "top_products": [
            {
                "rank": index,
                "id": product.get("id"),
                "title": product.get("title"),
                "category": product.get("category"),
                "dominant_color_name": product.get("dominant_color_name"),
                "image_similarity": product.get("image_similarity"),
                "category_score": product.get("category_score"),
                "color_score": product.get("color_score"),
                "text_score": product.get("text_score"),
                "style_score": product.get("style_score"),
                "text_style_score": product.get("text_style_score"),
                "category_penalty": product.get("category_penalty"),
                "color_penalty": product.get("color_penalty"),
                "final_score": product.get("final_score"),
                "match_reason": product.get("match_reason"),
                "query_style_label": product.get("query_style_label"),
                "product_styles": product.get("product_styles"),
            }
            for index, product in enumerate(products[:5], start=1)
        ],
    }


def post_query_image(client: httpx.Client, api_url: str, image_path: Path, top_k: int) -> tuple[dict, float]:
    content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    endpoint = f"{api_url.rstrip('/')}/api/search/upload"

    with image_path.open("rb") as image_file:
        start = time.perf_counter()
        response = client.post(
            endpoint,
            params={"top_k": top_k},
            files={"file": (image_path.name, image_file, content_type)},
        )
        latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    return response.json(), latency_ms


def evaluate_query(label: dict, client: httpx.Client, api_url: str, top_k: int) -> dict:
    query_id = str(label.get("query_id") or label.get("image") or "unknown")
    image_rel = label.get("image")
    image_path = None
    if image_rel:
        raw_image_path = Path(image_rel)
        image_path = raw_image_path if raw_image_path.is_absolute() else EVAL_DIR / raw_image_path
        image_path = image_path.resolve()
    expected_items = label.get("expected_items") or []

    base_result = {
        "query_id": query_id,
        "image": image_rel,
        "status": "pending",
        "latency_ms": None,
        "detected_count": 0,
        "detection_success": False,
        "expected_item_results": [],
        "detected_items": [],
        "error": None,
    }

    if not image_path or not image_path.exists():
        base_result["status"] = "missing_image"
        base_result["error"] = f"Missing query image: {image_rel}"
        base_result["expected_item_results"] = evaluate_expected_items(expected_items, detected_items=[])
        return base_result

    try:
        api_response, latency_ms = post_query_image(client, api_url, image_path, top_k=top_k)
    except Exception as exc:
        base_result["status"] = "api_error"
        base_result["error"] = str(exc)
        base_result["expected_item_results"] = evaluate_expected_items(expected_items, detected_items=[])
        return base_result

    detected_items = api_response.get("detected_items") or []
    base_result.update(
        {
            "status": "ok",
            "latency_ms": latency_ms,
            "detected_count": len(detected_items),
            "detection_success": len(detected_items) > 0,
            "detected_items": [summarize_detection_item(item) for item in detected_items],
        }
    )

    base_result["expected_item_results"] = evaluate_expected_items(expected_items, detected_items)

    return base_result


def evaluate_expected_items(expected_items: list[dict], detected_items: list[dict]) -> list[dict]:
    return [
        {
            "expected_index": expected_index,
            "type": expected_item.get("type"),
            "category_at_5": category_hit_at_5(expected_item, detected_items),
            "color_at_5": color_hit_at_5(expected_item, detected_items),
            "recall_at_5": recall_hit_at_5(expected_item, detected_items),
            "reciprocal_rank": reciprocal_rank(expected_item, detected_items),
        }
        for expected_index, expected_item in enumerate(expected_items)
    ]


def _ratio(values: list[bool]) -> float | None:
    if not values:
        return NO_VALUE
    return sum(1 for value in values if value) / len(values)


def compute_metrics(per_query_results: list[dict]) -> dict:
    total_queries = len(per_query_results)
    successful_api_queries = [result for result in per_query_results if result["status"] == "ok"]
    latencies = [result["latency_ms"] for result in successful_api_queries if result["latency_ms"] is not None]
    expected_results = [
        item_result
        for result in per_query_results
        for item_result in result.get("expected_item_results", [])
    ]

    category_values = [
        result["category_at_5"]
        for result in expected_results
        if result.get("category_at_5") is not NO_VALUE
    ]
    color_values = [
        result["color_at_5"]
        for result in expected_results
        if result.get("color_at_5") is not NO_VALUE
    ]
    recall_values = [
        result["recall_at_5"]
        for result in expected_results
        if result.get("recall_at_5") is not NO_VALUE
    ]
    reciprocal_ranks = [
        result["reciprocal_rank"]
        for result in expected_results
        if result.get("reciprocal_rank") is not NO_VALUE
    ]

    return {
        "detection_success_rate": (
            sum(1 for result in per_query_results if result["detection_success"]) / total_queries
            if total_queries
            else NO_VALUE
        ),
        "avg_detected_items": (
            mean(result["detected_count"] for result in per_query_results)
            if total_queries
            else NO_VALUE
        ),
        "category_at_5": _ratio(category_values),
        "color_at_5": _ratio(color_values),
        "recall_at_5": _ratio(recall_values),
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else NO_VALUE,
        "avg_latency_ms": mean(latencies) if latencies else NO_VALUE,
        "successful_api_queries": len(successful_api_queries),
        "failed_queries": total_queries - len(successful_api_queries),
    }


def write_csv(path: Path, per_query_results: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "status",
                "latency_ms",
                "detected_count",
                "detection_success",
                "category_at_5",
                "color_at_5",
                "recall_at_5",
                "reciprocal_rank",
                "error",
            ],
        )
        writer.writeheader()
        for result in per_query_results:
            expected_results = result.get("expected_item_results") or [{}]
            for expected in expected_results:
                writer.writerow(
                    {
                        "query_id": result["query_id"],
                        "status": result["status"],
                        "latency_ms": result["latency_ms"],
                        "detected_count": result["detected_count"],
                        "detection_success": result["detection_success"],
                        "category_at_5": expected.get("category_at_5"),
                        "color_at_5": expected.get("color_at_5"),
                        "recall_at_5": expected.get("recall_at_5"),
                        "reciprocal_rank": expected.get("reciprocal_rank"),
                        "error": result.get("error"),
                    }
                )


def print_summary(report: dict):
    print("\nEvaluation summary")
    print("==================")
    print(f"Run name: {report['run_name']}")
    print(f"API URL: {report['api_url']}")
    print(f"Total queries: {report['total_queries']}")
    for name, value in report["metrics"].items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Run visual search evaluation against the API.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--labels", default=str(EVAL_DIR / "labels.json"))
    parser.add_argument("--run-name", default="current")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    if isinstance(labels, dict):
        labels = [labels]
    if not isinstance(labels, list):
        raise ValueError("Labels file must contain a JSON list")

    with httpx.Client(timeout=args.timeout) as client:
        per_query_results = [
            evaluate_query(label, client=client, api_url=args.api_url, top_k=args.top_k)
            for label in labels
        ]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "api_url": args.api_url,
        "labels": str(labels_path),
        "total_queries": len(labels),
        "metrics": compute_metrics(per_query_results),
        "per_query_results": per_query_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_json = RESULTS_DIR / f"{args.run_name}.json"
    run_csv = RESULTS_DIR / f"{args.run_name}.csv"
    latest_json = RESULTS_DIR / "latest_eval.json"
    latest_csv = RESULTS_DIR / "latest_eval.csv"

    run_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(run_csv, per_query_results)
    shutil.copyfile(run_json, latest_json)
    shutil.copyfile(run_csv, latest_csv)

    print_summary(report)
    print(f"\nWrote: {run_json}")
    print(f"Wrote: {run_csv}")


if __name__ == "__main__":
    main()
