"""EXPLORATORY run of validation-only configurations on the test split.

The test split was already run once under the locked configuration, so anything measured here is
no longer an independent estimate: the configurations were proposed after looking at test results.
This script never touches `locked_config.json` or the locked test run; it writes to its own folder
and marks the snapshot `exploratory`. Report these numbers as post-hoc, next to the locked ones.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .catalog import CATALOG_PATH, Catalog
from .manifest import file_hash, load_manifest, manifest_path
from .pipeline import BASELINE, InProcessSystem, default_configs
from .run import MODEL_PATH, attach_target, failed_result, write_json

NOTE = ("EXPLORATORY: test split reused after the locked test run; configurations were chosen after "
        "viewing test results, so these numbers are post-hoc and not independent.")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    if (args.out / "results.json").exists():
        raise FileExistsError(args.out)
    labels = load_manifest(manifest_path("test"))
    all_configs = {name: replace(cfg, conf=args.conf) for name, cfg in default_configs().items()}
    names = [BASELINE, *[name for name in args.configs if name != BASELINE]]
    configs = {name: all_configs[name] for name in names}

    from AI.embedding_model import get_embedding_config

    catalog = Catalog.load(CATALOG_PATH)
    catalog.check_embedding_space(get_embedding_config().model_name, get_embedding_config().version)
    system = InProcessSystem(catalog, MODEL_PATH)
    system.warm_text_cache()
    queries = []
    for n, label in enumerate(labels, 1):
        pixels = label["path"].read_bytes()
        if hashlib.sha256(pixels).hexdigest() != label["sha256"]:
            raise ValueError(f"Image changed: {label['query_id']}")
        query = {key: value for key, value in label.items() if key != "path"}
        query["results"] = {}
        for name, config in configs.items():
            try:
                result = system.run_query(pixels, config)
            except Exception as exc:  # keep the query, score it as a miss
                result = failed_result(config, exc)
            result["target"] = attach_target(result, label)
            query["results"][name] = result
        queries.append(query)
        print(f"[{n}/{len(labels)}] {label['query_id']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "results.json", {
        "exploratory": True, "note": NOTE, "stage": "test", "conf": args.conf,
        "manifest_sha256": file_hash(manifest_path("test")), "catalog_sha256": catalog.sha256,
        "configs": {name: asdict(cfg) for name, cfg in configs.items()}, "queries": queries})
    print(args.out / "results.json")


if __name__ == "__main__":
    main()
