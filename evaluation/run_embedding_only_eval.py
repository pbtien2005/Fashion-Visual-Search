import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI.vectorizer import extract_vector
from evaluation.run_eval import compute_metrics, evaluate_expected_items, print_summary


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def dot(left, right) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def serialize_product(item: dict, similarity: float) -> dict:
    return {
        "id": item.get("id") or item.get("asin"),
        "title": item.get("title"),
        "brand": item.get("brand"),
        "price": item.get("price"),
        "category": str(item.get("category") or ""),
        "image_url": item.get("r2_image_url") or item.get("thumbnailImage"),
        "dominant_color_name": item.get("dominant_color_name"),
        "dominant_color_rgb": item.get("dominant_color_rgb"),
        "similarity": similarity,
        "image_similarity": similarity,
        "final_score": similarity,
        "match_reason": "embedding similarity only",
    }


def top_products(query_vector: list[float], products: list[dict], top_k: int) -> list[dict]:
    scored = []
    for item in products:
        embedding = item.get("embedding")
        if not embedding or len(embedding) != len(query_vector):
            continue
        scored.append((dot(query_vector, embedding), item))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [serialize_product(item, score) for score, item in scored[:top_k]]


def summarize_products(products: list[dict]) -> list[dict]:
    return [
        {
            "rank": index,
            "id": product.get("id"),
            "title": product.get("title"),
            "category": product.get("category"),
            "dominant_color_name": product.get("dominant_color_name"),
            "image_similarity": product.get("image_similarity"),
            "final_score": product.get("final_score"),
            "match_reason": product.get("match_reason"),
        }
        for index, product in enumerate(products, start=1)
    ]


def evaluate_label(label: dict, products: list[dict], top_k: int) -> dict:
    query_id = str(label.get("query_id") or label.get("image") or "unknown")
    image_path = EVAL_DIR / label["image"]
    result = {
        "query_id": query_id,
        "image": label.get("image"),
        "status": "pending",
        "latency_ms": None,
        "detected_count": 0,
        "detection_success": False,
        "expected_item_results": [],
        "detected_items": [],
        "error": None,
    }

    start = time.perf_counter()
    try:
        image = Image.open(image_path).convert("RGB")
        query_vector = extract_vector(image)
        if query_vector is None:
            raise RuntimeError("Could not extract query embedding")

        products_top = top_products(query_vector, products=products, top_k=top_k)
        detected_items = [
            {
                "class_name": "whole_image_embedding",
                "class_name_vi": "Whole image embedding",
                "confidence": None,
                "bbox": None,
                "is_confident_match": True,
                "similar_products": products_top,
                "top_products": summarize_products(products_top),
            }
        ]
        result.update(
            {
                "status": "ok",
                "latency_ms": (time.perf_counter() - start) * 1000,
                "detected_count": 1,
                "detection_success": True,
                "detected_items": detected_items,
                "expected_item_results": evaluate_expected_items(
                    label.get("expected_items") or [],
                    detected_items,
                ),
            }
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["expected_item_results"] = evaluate_expected_items(label.get("expected_items") or [], [])

    return result


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
            for expected in result.get("expected_item_results") or [{}]:
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure whole-image embedding retrieval.")
    parser.add_argument("--labels", default=str(EVAL_DIR / "labels.real.json"))
    parser.add_argument("--catalog", default="data/fashion_with_vectors.json")
    parser.add_argument("--run-name", default="embedding_only_whole_image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    catalog_path = Path(args.catalog)
    labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    products = [item for item in catalog if item.get("embedding")]

    print(f"Loaded products with embeddings: {len(products)}")
    per_query_results = [
        evaluate_label(label, products=products, top_k=args.top_k)
        for label in labels
    ]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "api_url": None,
        "labels": str(labels_path),
        "catalog": str(catalog_path),
        "mode": "whole query image embedding only; no detection; no rerank",
        "total_queries": len(labels),
        "metrics": compute_metrics(per_query_results),
        "per_query_results": per_query_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_json = RESULTS_DIR / f"{args.run_name}.json"
    run_csv = RESULTS_DIR / f"{args.run_name}.csv"
    run_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(run_csv, per_query_results)

    print_summary(report)
    print(f"\nWrote: {run_json}")
    print(f"Wrote: {run_csv}")


if __name__ == "__main__":
    main()
