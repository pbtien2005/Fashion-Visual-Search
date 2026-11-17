"""Run benchmark inference and reserve each scored split before model use."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .catalog import CATALOG_PATH, Catalog
from .manifest import (
    file_hash, load_manifest, manifest_path, portable_path, reject_old_outfits, require_sample_size,
    resolve_path, validate_splits,
)
from .metrics import match_target
from .pipeline import BASELINE, GAMMA, VALIDATION_ONLY, WHOLE, InProcessSystem, default_configs, new_timings

MODEL_PATH = Path("models/object-detection/best.pt")
PARTIAL = "queries.partial.jsonl"
RUN_IDENTITY = ("stage", "manifest_sha256", "model_sha256", "catalog_sha256", "conf", "crop_config", "configs")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def metadata():
    packages = {}
    for name in ("torch", "ultralytics", "transformers", "torchvision", "opencv-python-headless"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "commit": commit.stdout.strip() or None,
            "python": platform.python_version(), "platform": platform.platform(), "versions": packages}


def failed_result(config, exc):
    """A query that raised still counts: an empty result is a miss, not a dropped sample."""
    return {
        "geometry": None, "quality": {"gamma_applied": None}, "detections": [],
        "timings_ms": new_timings(None), "error": f"{type(exc).__name__}: {exc}",
        "detection_measured": getattr(config, "kind", None) != "whole_image",
    }


def attach_target(result, label):
    if result["geometry"] is None:
        return {"index": None, "iou": 0.0, "best_iou_any_class": 0.0, "bbox": None}
    width, height = (result["geometry"][axis] for axis in ("width", "height"))
    bbox = [value * size for value, size in zip(label["bbox"], (width, height) * 2, strict=True)]
    target = match_target(result["detections"], label["class_name"], bbox)
    if not result.get("detection_measured", True):
        target = {"index": 0, "iou": None, "best_iou_any_class": None}
    return {**target, "bbox": bbox}


def read_partial(path, identity):
    """Return queries already inferred by an interrupted run with the same configuration."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # the process died while writing this line
    if not rows or rows[0].get("identity") != json.loads(json.dumps(identity)):
        raise ValueError("Interrupted run used a different configuration; it cannot be resumed")
    return rows[1:]


def verify_test_lock(lock, snapshot, labels, definitions):
    for key in ("model_sha256", "catalog_sha256", "conf", "crop_config"):
        if lock.get(key) != snapshot[key]:
            raise ValueError(f"Test configuration differs from validation lock: {key}")
    overlap = set(lock["validation_images"]) & {label["sha256"] for label in labels}
    if overlap:
        ids = [label["query_id"] for label in labels if label["sha256"] in overlap]
        raise ValueError(f"Test image also in locked validation: {', '.join(ids)}")
    for name, definition in definitions.items():
        source = lock.get("C3_source") if name == "C3_selected" else name
        expected = lock["configs"].get(source)
        if expected is not None and name == "C3_selected":
            expected = {**expected, "name": name}
        if expected != definition:
            raise ValueError(f"Test configuration differs from validation lock: {name}")
    if "C3_selected" in definitions and lock.get("C3_selected") != definitions["C3_selected"]:
        raise ValueError("C3_selected differs from validation lock")


def planned_configs(args, lock):
    configs = {name: replace(config, conf=args.conf) for name, config in default_configs().items()}
    if args.stage == "robustness":
        return {name: configs[name] for name in (WHOLE, BASELINE, GAMMA)}
    if args.stage == "test":
        if lock["C3_source"] not in (*VALIDATION_ONLY, BASELINE):
            raise ValueError("Invalid C3_source in validation lock")
        selected = configs[lock["C3_source"]]
        configs = {name: config for name, config in configs.items() if name not in VALIDATION_ONLY}
        if selected.name != BASELINE:
            configs["C3_selected"] = replace(selected, name="C3_selected")
    return configs


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test", "robustness"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--smoke", type=int)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run; finished queries are not inferred again")
    parser.add_argument("--api-url")
    parser.add_argument("--config-name")
    return parser


def execute(args):
    if args.smoke is not None and (args.stage != "validation" or args.smoke < 1):
        raise ValueError("--smoke N requires validation and a positive N")
    if bool(args.api_url) != bool(args.config_name) or (args.api_url and args.stage == "robustness"):
        raise ValueError("--api-url and --config-name must be paired (validation or test)")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be in [0, 1]")

    folder = args.run_dir / ("smoke" if args.smoke is not None else args.stage)
    partial = folder / PARTIAL
    resuming = args.smoke is None and args.resume and partial.exists() and not (folder / "results.json").exists()
    if args.smoke is None and folder.exists() and not resuming:
        hint = " (interrupted; continue with --resume)" if partial.exists() else ""
        raise FileExistsError(f"Run already reserved: {folder}{hint}")
    lock = read_json(args.run_dir / "locked_config.json") if args.stage == "test" else None
    labels_path = args.labels or manifest_path("perturbed" if args.stage == "robustness" else args.stage)
    manifest_sha256 = file_hash(labels_path)

    if args.stage == "robustness":
        from .build_perturbed import load_perturbed, write_robustness_report

        labels = load_perturbed(labels_path)
    else:
        labels = load_manifest(labels_path)
        if args.stage == "test":
            reject_old_outfits(labels)
            validation_manifest = resolve_path(lock["validation_manifest"])
            validate_splits(load_manifest(validation_manifest), labels)
            if file_hash(validation_manifest) != lock["validation_manifest_sha256"]:
                raise ValueError("Validation manifest differs from lock")
        elif args.smoke is None:
            test_labels = load_manifest(manifest_path("test"))
            reject_old_outfits(test_labels)
            validate_splits(labels, test_labels)
        if args.smoke is None:
            require_sample_size(args.stage, labels)
        else:
            labels = labels[:args.smoke]

    # Check hashes before building the index or calling either model.
    model_sha256 = file_hash(args.model)
    catalog_sha256 = file_hash(args.catalog)
    if lock is not None:
        for key, actual in (("model_sha256", model_sha256), ("catalog_sha256", catalog_sha256)):
            if lock.get(key) != actual:
                raise ValueError(f"Test configuration differs from validation lock: {key}")

    embedding = None
    if args.api_url:
        from .http_adapter import HttpSystem

        system = HttpSystem(args.api_url)
        configs = {args.config_name: None}
        definitions = {args.config_name: {"name": args.config_name, "kind": "http", "api_url": args.api_url}}
        crop_config = None
    else:
        from AI.embedding_model import get_embedding_config

        configs = planned_configs(args, lock)
        definitions = {name: asdict(cfg) for name, cfg in configs.items()}
        catalog = Catalog.load(args.catalog)
        if catalog.sha256 != catalog_sha256:
            raise ValueError("Catalog changed while loading")
        config = get_embedding_config()
        catalog.check_embedding_space(config.model_name, config.version)
        embedding = asdict(config)
        system = InProcessSystem(catalog, args.model)
        crop_config = system.crop_config

    snapshot = {
        **metadata(), "stage": args.stage, "smoke": args.smoke,
        "manifest": portable_path(labels_path), "manifest_sha256": manifest_sha256,
        "model": portable_path(args.model), "model_sha256": model_sha256,
        "catalog": portable_path(args.catalog), "catalog_sha256": catalog_sha256,
        "conf": args.conf, "crop_config": crop_config, "configs": definitions, "embedding": embedding,
    }
    if lock is not None:
        verify_test_lock(lock, snapshot, labels, definitions)
    identity = {key: snapshot[key] for key in RUN_IDENTITY}
    finished = {}
    if resuming:
        finished = {query["query_id"]: query for query in read_partial(partial, identity)}
        print(f"Resuming: {len(finished)} queries already inferred", flush=True)
    else:
        folder.mkdir(parents=True, exist_ok=args.smoke is not None)
        (folder / "results.json").unlink(missing_ok=True)
    # Rewriting drops a line cut off by the interruption.
    partial.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                               for row in ({"identity": identity}, *finished.values())), encoding="utf-8")

    snapshot["resumed_after"] = len(finished) if resuming else None
    queries, errors = [], []
    # A configuration that never succeeded points at a broken setup, so its first failure aborts.
    proven = set(configs) if finished else set()
    try:
        if any(getattr(config, "kind", None) == "rerank" for config in configs.values()):
            snapshot["text_cache_warmup_ms"] = system.warm_text_cache()
        with partial.open("a", encoding="utf-8") as stream:
            for n, label in enumerate(labels, 1):
                query = finished.get(label["query_id"])
                if query is not None:
                    if query["sha256"] != label["sha256"]:
                        raise ValueError(f"Query image changed since the interrupted run: {label['query_id']}")
                    queries.append(query)
                    continue
                pixels = label["path"].read_bytes()
                if hashlib.sha256(pixels).hexdigest() != label["sha256"]:
                    raise ValueError(f"Query image changed after loading manifest: {label['query_id']}")
                query = {key: value for key, value in label.items() if key != "path"}
                query["results"] = {}
                for name, config in configs.items():
                    try:
                        result = (system.run_query(pixels, label["path"].name) if config is None
                                  else system.run_query(pixels, config))
                        proven.add(name)
                    except Exception as exc:
                        if name not in proven:
                            raise
                        result = failed_result(config, exc)
                        errors.append(f"{label['query_id']} {name}: {result['error']}")
                        print(f"  ! {errors[-1]}", flush=True)
                    if args.stage != "robustness":
                        result["target"] = attach_target(result, label)
                    query["results"][name] = result
                stream.write(json.dumps(query, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                queries.append(query)
                print(f"[{n}/{len(labels)}] {label['query_id']}", flush=True)
    except BaseException:
        if not queries and not finished:
            # Nothing was scored, so release the reservation for a clean rerun.
            partial.unlink(missing_ok=True)
            if args.smoke is None:
                try:
                    folder.rmdir()
                except OSError:
                    pass
        raise

    snapshot["queries"] = queries
    snapshot["errors"] = errors
    write_json(folder / "results.json", snapshot)
    partial.unlink()
    if args.stage == "robustness":
        write_robustness_report(folder, snapshot)
    print(folder / "results.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = make_parser()
    args = parser.parse_args()
    try:
        execute(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

