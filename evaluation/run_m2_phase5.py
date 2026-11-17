"""Phase5: init protocol -> validation/lock -> regression test, never tune on test."""

import argparse
from dataclasses import asdict
from pathlib import Path
import os

from AI.hsv_descriptor import HSVConfig
from evaluation.m2_annotations import ROOT, load_manifest, read_json, write_json, file_hash
from evaluation.m2_phase5 import (CRITERIA, MODES, assess_validation, audit_cache, digest, frozen_color_rows,
                                 input_identity, summarize_color, verify_lock)
from evaluation.run_m2_eval import save_report
from object_detection.segmentation import MaskConfig


def create_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        import json
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "validation", "test"))
    parser.add_argument("--manifest", default="evaluation/m2/annotations.review2.json")
    parser.add_argument("--base-report", default="docs/m2/phase4/review2/results.json")
    parser.add_argument("--output", default="docs/m2/phase5")
    args = parser.parse_args()
    output = Path(args.output)
    identity = input_identity(args.manifest, args.base_report)
    fingerprint = digest(identity)
    manifest = load_manifest(args.manifest)
    base = read_json(args.base_report)
    if (base["manifest_sha256"] != identity["manifest_sha256"] or base["model_sha256"] != identity["model_sha256"]
            or base["HSV_version"] != HSVConfig().version or base["preprocess_config"] != manifest["preprocess_config"]):
        raise ValueError("Phase4 source/model/manifest/profile differs from phase5 input")
    for mode in MODES:
        if base["mask_configs"][mode] != asdict(MaskConfig(mode=mode)):
            raise ValueError("This short experiment freezes existing gates, not a new parameter sweep")
    if args.stage == "init":
        protocol = {"schema_version": "m2.phase5.protocol.v1", "input_fingerprint": fingerprint,
                    "inputs": identity, "mask_modes": list(MODES), "selection_criteria": CRITERIA,
                    "color_metric": "HSV147 Hellinger similarity against reviewed clothing-reference pixels; miss=0",
                    "comparison": {"H0-M1": "off", "B2": "raw", "B3": "open_close"},
                    "prior_test_exposure": True, "independent_test": False,
                    "reuse_detection": "Frozen genuine phase4 YOLO boxes/IDs; no new inference, encoder or candidate search",
                    "limitations": "Shared 20-image regression; test masks/IoU already observed in phase4. Selection code filters validation only."}
        protocol["lock_sha256"] = digest(protocol)
        create_json(output / "protocol.json", protocol)
        create_json(output / "cache_audit.json", audit_cache())
        print("Protocol saved before color validation; product retrieval is not measured")
        return 0
    protocol = read_json(output / "protocol.json")
    verify_lock(protocol, fingerprint)
    if args.stage == "test":
        lock = read_json(output / "locked_config.json")
        verify_lock(lock, fingerprint)
        if file_hash(output / "validation/results.json") != lock["validation_results_sha256"]:
            raise ValueError("Validation artifact changed after lock")
    folder = output / args.stage
    if folder.exists():
        raise ValueError("Run output exists; keep prior evidence and use a new run directory")
    # This subprocess uses the shared production profile and strict color failures.
    # Do not alter user .env or the running services.
    os.environ["HSV_CONFIG_PATH"] = str(ROOT / "docs/m3_contract/hsv147.json")
    os.environ["HSV_MISSING_POLICY"] = "error"
    rows = frozen_color_rows(manifest, base, args.stage)
    report = {"schema_version": "m2.phase5.ablation.v1", "rows": rows, "summary": summarize_color(rows),
              "split": args.stage, "input_fingerprint": fingerprint, "protocol_sha256": file_hash(output / "protocol.json"),
              "genuine_detections_reused": True, "independent_test": False,
              "product_retrieval": "not_measured", "default_selection_uses": "validation_only"}
    folder.mkdir(parents=True)
    save_report(report, folder)
    if args.stage == "validation":
        decision = assess_validation(rows)
        lock = {"schema_version": "m2.phase5.lock.v1", "input_fingerprint": fingerprint,
                "mask_modes": list(MODES), "mask_configs": {m: asdict(MaskConfig(mode=m)) for m in MODES},
                "decision": decision, "protocol_sha256": file_hash(output / "protocol.json"),
                "validation_results_sha256": file_hash(folder / "results.json"),
                "deployment_mask_config": asdict(MaskConfig(mode=decision["deployment_mode"])),
                "prior_test_exposure": True, "independent_test": False}
        lock["lock_sha256"] = digest(lock)
        create_json(output / "locked_config.json", lock)
        create_json(output / "mask_config.final.json", lock["deployment_mask_config"])
        print(f"Validation complete; locked deployment mode={decision['deployment_mode']}; test not used for selection")
    else:
        print("Regression test complete under unchanged lock; no config tuning or product-retrieval claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
