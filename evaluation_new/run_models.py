"""Run the existing benchmark pipeline once per embedding model.

Kept from evaluation/benchmark (imported, not copied): manifests, M1 preparation, the
ClothingDetector crop code (tight/padded/wide, 224 px, white pad), configuration definitions
(A0, C2, B0, C3_selected = C3_w1 as locked on validation), production rerank, thresholds,
target matching and scoring.

Changed per model only:
  * catalog vectors: every product re-embedded with the model from the same pixels as the stored
    FashionCLIP vector (padded YOLO product crop for 1709/1752, whole photo otherwise; see
    catalog_crops.py, verified at cosine >= 0.998 against the stored vectors);
  * query vectors: the same model on each crop / whole image;
  * text/style vectors inside the B0 rerank: the model's own text encoder.
YOLO boxes are replayed from evaluation_new/cache/detections.json (made in Docker, see
detect_cache.py). Product HSV colours stay keyed to FashionCLIP: colour does not depend on the
embedding model, so only the vectors differ between runs.

    python -m evaluation_new.run_models --models all --splits test validation
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # evaluation/benchmark resolves data paths from the repo root
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from evaluation_new.catalog_crops import embedding_image  # noqa: E402

CACHE = ROOT / "evaluation_new/cache"
# Product HSV colours: the cache rebuilt for all 1752 products (build_color_cache.py). .env points
# to data/color_cache_hsv147.json, which covers only the 28-product sample; BENCHMARK.md requires a
# full cache, because rerank_products puts the colour name in the text/style prompt.
FULL_COLOR_CACHE = CACHE / "color_cache_hsv147_full.json"
if "HSV_COLOR_CACHE_PATH" not in os.environ and FULL_COLOR_CACHE.exists():
    os.environ["HSV_COLOR_CACHE_PATH"] = str(FULL_COLOR_CACHE)

# Same .env as the old runner (mask mode, text/style switch); set variables win over .env.
load_dotenv(ROOT / ".env")

# Overridable only for a smoke test on the 28-product sample catalog.
RESULTS = ROOT / os.getenv("BENCH_RESULTS", "evaluation_new/results")
YOLO_PATH = ROOT / "object_detection/best.pt"
CATALOG_PATH = ROOT / os.getenv("BENCH_CATALOG", "data/fashion_with_vectors.json")
FASHIONCLIP_ID, FASHIONCLIP_VERSION = "patrickjohncyh/fashion-clip", 2
STORED = "fashionclip_stored"  # harness check: production code and the catalog's stored vectors
SPLITS = ("test", "validation")
TEXT_FAILURES = {"batches": 0}  # text-encoder errors that made production skip text/style
TEXT_CONFIGS = ("B0_current", "C3_selected")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- detection replay
class _Box:
    def __init__(self, raw):
        self.xyxy, self.cls, self.conf = [raw["xyxy"]], [raw["cls"]], [raw["conf"]]


class _Result:
    def __init__(self, boxes):
        self.boxes = [_Box(raw) for raw in boxes]


class ReplayYolo:
    """Stands in for ultralytics.YOLO inside ClothingDetector; returns the boxes saved in Docker.

    The runner sets `current` to the query file's SHA-256 before each run_query. The prepared
    array must have the size YOLO saw; any pixel drift (different JPEG decoder) is counted.
    """

    def __init__(self, path=CACHE / "detections.json"):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["model_sha256"] != sha256_file(YOLO_PATH):
            raise ValueError("detections.json was made with a different YOLO checkpoint")
        self.meta = {key: data[key] for key in ("model_sha256", "conf", "versions", "created_at")}
        self.names = {int(key): value for key, value in data["names"].items()}
        self.entries = data["entries"]
        self.current = None
        self.pixel_drift = set()

    def predict(self, bgr, conf, verbose=False):
        entry = self.entries.get(f"{self.current}:{conf}")
        if entry is None:
            raise KeyError(f"No cached YOLO boxes for image {self.current} at conf={conf}; rerun detect_cache.py")
        if (bgr.shape[1], bgr.shape[0]) != (entry["width"], entry["height"]):
            raise ValueError(f"{entry['query_id']}: prepared size differs from the cached YOLO input")
        if hashlib.sha256(bgr.tobytes()).hexdigest() != entry["detection_sha256"]:
            self.pixel_drift.add(entry["query_id"])
        return [_Result(entry["boxes"])]


# ---------------------------------------------------------------- catalog
def catalog_rows():
    from evaluation.benchmark.catalog import Catalog

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Thiếu {CATALOG_PATH.relative_to(ROOT)} (tải từ Google Drive của nhóm)")
    reference = Catalog.load(CATALOG_PATH)  # same validity filter as the old benchmark
    raw = {str(row["id"]): row for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))}
    return reference, [raw[product["id"]] for product in reference.products]


def catalog_vectors(backend, rows):
    """Embed every catalog photo once per model; cached as .npz keyed by model and image hashes."""
    image_dir = CACHE / "catalog_images"
    paths = [image_dir / f"{row['id']}.img" for row in rows]
    missing = [row["id"] for row, path in zip(rows, paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh catalog (vd {missing[:3]}); chạy fetch_catalog_images")
    # Images + which pixels each stored vector came from (crop box or whole photo).
    crop_keys = "".join(f"{row.get('embedding_source')}{row.get('product_crop_bbox')}" for row in rows)
    fingerprint = hashlib.sha256(("".join(sha256_file(p) for p in paths) + crop_keys).encode()).hexdigest()
    out = CACHE / "embeddings" / f"{backend.spec.key}-{len(rows)}.npz"
    if out.exists():
        cached = np.load(out, allow_pickle=False)
        if str(cached["fingerprint"]) == fingerprint and list(cached["ids"]) == [r["id"] for r in rows]:
            return cached["vectors"], float(cached["seconds"])
    started = time.perf_counter()
    vectors = []
    for i in range(0, len(paths), 64):
        images = [embedding_image(row, p.read_bytes()) for row, p in zip(rows[i:i + 64], paths[i:i + 64])]
        vectors.append(backend.encode_images(images))
        print(f"  catalog {min(i + 64, len(paths))}/{len(paths)}", end="\r", flush=True)
    vectors = np.concatenate(vectors).astype(np.float32)
    seconds = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, vectors=vectors, ids=np.array([r["id"] for r in rows]), fingerprint=fingerprint,
             seconds=seconds)
    print(f"  catalog {len(paths)} ảnh, {seconds:.1f}s")
    return vectors, seconds


# ---------------------------------------------------------------- patching
def use_model_for_text_and_colour(backend):
    """Route the production text encoder to `backend`; keep product colours on FashionCLIP."""
    import AI.embedding_model as embedding_model
    import api.search_service as search_service
    from AI import production_color

    if not hasattr(search_service, "_bench_original"):
        search_service._bench_original = (search_service.encode_texts, search_service.product_color)
    original_texts, original_colour = search_service._bench_original
    encode = original_texts if backend is None else backend.encode_texts

    def counted(texts, model_name=None):
        # Production catches text-encoder errors and drops text/style for the whole batch
        # (search_service.compute_text_style_scores). Count those batches per run.
        try:
            return encode(texts, model_name=model_name)
        except Exception:
            TEXT_FAILURES["batches"] += 1
            raise

    search_service.encode_texts = counted
    embedding_model.encode_texts = counted
    search_service.product_color = (
        lambda product_id, model_name, model_version, store=None:
        production_color.product_color(product_id, FASHIONCLIP_ID, FASHIONCLIP_VERSION, store))
    search_service._style_prompt_vector_cache.clear()


# ---------------------------------------------------------------- runs
def bench_configs(has_text):
    from evaluation.benchmark.pipeline import BASELINE, EMBEDDING, WHOLE, default_configs

    locked = json.loads((ROOT / "docs/benchmark/main/locked_config.json").read_text(encoding="utf-8"))
    configs = default_configs()
    chosen = {WHOLE: configs[WHOLE], EMBEDDING: configs[EMBEDDING]}
    if has_text:
        chosen[BASELINE] = configs[BASELINE]
        chosen["C3_selected"] = replace(configs[locked["C3_source"]], name="C3_selected")
    return chosen


def versions():
    out = {}
    for name in ("torch", "transformers", "open_clip_torch", "timm", "numpy"):
        try:
            out[name] = version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def run_split(model_key, split, system, configs, extra, force=False):
    from evaluation.benchmark.manifest import load_manifest, manifest_path
    from evaluation.benchmark.run import attach_target, failed_result, write_json

    folder = RESULTS / model_key / split
    if (folder / "results.json").exists() and not force:
        print(f"  {split}: đã có kết quả, bỏ qua")
        return folder / "results.json"
    folder.mkdir(parents=True, exist_ok=True)
    labels = load_manifest(manifest_path(split))
    TEXT_FAILURES["batches"] = 0
    queries, errors = [], []
    for n, label in enumerate(labels, 1):
        pixels = label["path"].read_bytes()
        if hashlib.sha256(pixels).hexdigest() != label["sha256"]:
            raise ValueError(f"Ảnh truy vấn đổi so với manifest: {label['query_id']}")
        query = {key: value for key, value in label.items() if key != "path"}
        query["results"] = {}
        system.detector._model.current = label["sha256"]
        for name, config in configs.items():
            try:
                result = system.run_query(pixels, config)
            except Exception as exc:  # noqa: BLE001 - counted as a miss, like the old runner
                result = failed_result(config, exc)
                errors.append(f"{label['query_id']} {name}: {result['error']}")
            result["target"] = attach_target(result, label)
            query["results"][name] = result
        queries.append(query)
        print(f"  {split} [{n}/{len(labels)}] {label['query_id']}", end="\r", flush=True)
    print()
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "versions": versions(), "stage": split, "model_key": model_key,
        "manifest_sha256": sha256_file(manifest_path(split)), "conf": 0.5,
        "configs": {name: asdict(cfg) for name, cfg in configs.items()},
        **extra, "yolo_pixel_drift": sorted(system.detector._model.pixel_drift),
        "text_style_skipped_batches": TEXT_FAILURES["batches"],
        "queries": queries, "errors": errors,
    }
    if errors:
        print(f"  ! {len(errors)} lỗi suy luận (tính là trượt), vd: {errors[0]}")
    write_json(folder / "results.json", snapshot)
    return folder / "results.json"


def run_model(model_key, splits=SPLITS, force=False):
    import torch

    from evaluation.benchmark.catalog import Catalog
    from evaluation.benchmark.pipeline import InProcessSystem

    from .models import DEVICE, MODELS, load_backend

    if all((RESULTS / model_key / split / "results.json").exists() for split in splits) and not force:
        print(f"{model_key}: đã đủ kết quả")
        return
    print(f"== {model_key}")
    reference, rows = catalog_rows()
    replay = ReplayYolo()
    if model_key == STORED:
        backend, catalog, embed_fn, seconds = None, reference, None, None
        spec = {"key": STORED, "hf_id": FASHIONCLIP_ID, "backend": "production", "group": "check",
                "label": "FashionCLIP, vector gốc"}
        has_text = True
    else:
        backend = load_backend(model_key)
        vectors, seconds = catalog_vectors(backend, rows)
        catalog = Catalog([{**row, "embedding": vector.tolist(), "embedding_model": MODELS[model_key].hf_id}
                           for row, vector in zip(rows, vectors)], dimension=vectors.shape[1])
        embed_fn, spec, has_text = backend.encode_image, asdict(MODELS[model_key]), backend.has_text
    use_model_for_text_and_colour(backend)
    os.environ["EMBEDDING_MODEL"] = spec["hf_id"]
    system = InProcessSystem(catalog, YOLO_PATH, embed_fn=embed_fn)
    system.detector._model = replay
    configs = bench_configs(has_text)
    extra = {"model": spec, "device": DEVICE if backend else "production", "dimension": catalog.dimension,
             "catalog_products": len(catalog.products), "catalog_sha256": reference.sha256,
             "catalog_embed_seconds": seconds, "yolo": replay.meta,
             "color_cache": {"path": os.environ["HSV_COLOR_CACHE_PATH"],
                             "sha256": sha256_file(os.environ["HSV_COLOR_CACHE_PATH"])}}
    if has_text:
        extra["text_cache_warmup_ms"] = system.warm_text_cache()
    for split in splits:
        run_split(model_key, split, system, configs, extra, force)
    del system, backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    from .models import MODELS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"],
                        help=f"all | {STORED} | " + " | ".join(MODELS))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--force", action="store_true", help="chạy lại cả khi đã có results.json")
    args = parser.parse_args()
    keys = [STORED, *MODELS] if args.models == ["all"] else args.models
    for key in keys:
        run_model(key, tuple(args.splits), args.force)


if __name__ == "__main__":
    main()
