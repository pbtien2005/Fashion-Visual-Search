"""Build/check a local M2 snapshot with explicit inputs, checksums and no weights/secrets.

Uses only the standard library. Never commits, pushes, deletes directories, or
copies .env/Git/model weights. Existing output must carry our package marker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NAME = "M2_SHARED_20261114"
OUTPUT = ROOT / "output/m2/handoff"
ALLOWED = {".py", ".md", ".json", ".csv", ".txt", ".html", ".svg", ".png", ".jpg", ".jpeg", ".toml", ".yml", ".yaml", ".sql"}
EXCLUDED = {".git", ".env", ".venv", "venv", "__pycache__", "node_modules", ".hf_cache", ".huggingface"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def selected_files():
    files = {}
    for directory in ("AI", "api", "db", "etl", "object_detection", "evaluation", "tests", "docs/m2", "docs/m3_contract", "docs/data_sources", "docs/m3_results", "docs/m3_handoff_evidence", "docs/m3_production_evidence"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or set(path.relative_to(ROOT).parts) & EXCLUDED:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(("evaluation/queries/", "evaluation/results/")):
                continue
            if path.suffix.lower() in ALLOWED or path.name in {"Dockerfile", "Dockerfile.tests", ".dockerignore"}:
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Refusing a source file outside the workspace")
                files[relative] = path
    for path in (ROOT / "docs").glob("*.md"):
        files[path.relative_to(ROOT).as_posix()] = path
    for relative in ("main.py", "searcher.py", "requirements.txt", "docker-compose.yml", "AGENTS.md", ".env.example", ".gitignore", "README.md",
                     "docs/m3_results/catalog_color_cache.json", "docs/m3_results/catalog_manifest.json"):
        path = ROOT / relative
        if path.is_file():
            files[relative] = path
    manifest = load(ROOT / "evaluation/m2/annotations.review2.json")
    for row in manifest["queries"]:
        path = ROOT / row["image"]
        if sha(path) != row["sha256"]:
            raise ValueError("Evaluation image checksum mismatch")
        files[row["image"]] = path
    # A regression manifest and M1 CSVs also name one old development-only image.
    for row in load(ROOT / "evaluation/m2/regression_manifest.json")["queries"]:
        files[row["image"]] = ROOT / row["image"]
    return files, manifest


def add_samples(bundle, manifest):
    folder = bundle / "samples"
    folder.mkdir(exist_ok=True)
    rows = [row for row in manifest["queries"] if row["mask_status"] == "human_reviewed"]
    if len(rows) != 5:
        raise ValueError("This handoff expects exactly five reviewed reference masks")
    records = []
    attribution = ["# Nguồn ảnh của 5 mẫu minh họa", "", "Ảnh mẫu giữ pixel file gốc; reference PNG nằm trên canvas M1 sau EXIF/resize. Overlay là ảnh dẫn xuất phục vụ đánh giá, giữ nguồn AI gợi ý/người kiểm. Không coi mẫu này là test độc lập.", "",
                   "| Mẫu | Tác giả | Giấy phép | Nguồn / license |", "|---|---|---|---|"]
    for row in rows:
        if any(symbol in row["query_id"] for symbol in ("/", "\\", ":")) or row["query_id"] in {".", ".."}:
            raise ValueError("Unsafe sample file name")
        image_name = row["query_id"] + Path(row["image"]).suffix.lower()
        mask_name = row["query_id"] + ".reference.png"
        shutil.copyfile(ROOT / row["image"], folder / image_name)
        shutil.copyfile(ROOT / row["mask_path"], folder / mask_name)
        records.append(dict(row, sample_image=image_name, sample_mask=mask_name))
        links = f"[ảnh]({row['source_url']})" + (f" / [license]({row['license_url']})" if row.get("license_url") else "")
        attribution.append(f"| {row['query_id']} | {row['author']} | {row['license']} | {links} |")
    (folder / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n", encoding="utf-8")
    write(folder / "manifest.json", {"preprocess_config": manifest["preprocess_config"], "mask_coordinate_space": "m1_working_image", "samples": records})
    all_sources = ["# Nguồn dữ liệu đánh giá dùng chung", "", "20 target từ M1, có người dùng review; ảnh không tự trở thành 20 ảnh đóng góp mới Người2. Giữ nguồn/giấy phép khi ghép bài. File manifest trong evaluation/m2/ chứa hash, split, geometry, nguồn gợi ý và người kiểm.", "",
                   "| Query | Tác giả | Giấy phép | Nguồn |", "|---|---|---|---|"]
    for row in manifest["queries"]:
        license_link = f"[{row['license']}]({row['license_url']})" if row.get("license_url") else row["license"]
        all_sources.append(f"| {row['query_id']} | {row['author']} | {license_link} | [ảnh]({row['source_url']}) |")
    (bundle / "DATA_ATTRIBUTION.md").write_text("\n".join(all_sources) + "\n", encoding="utf-8")


def check(bundle):
    bundle = Path(bundle).resolve()
    manifest = load(bundle / "package_manifest.json")
    if manifest.get("schema") != "m2.handoff.package.v1":
        raise ValueError("Not an owned M2 package")
    for relative, expected in manifest["files_sha256"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or set(Path(relative).parts) & EXCLUDED or path.suffix.lower() in {".pt", ".onnx", ".safetensors"}:
            raise ValueError("Unsafe path or forbidden secret/model file")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(manifest["files_sha256"]) | {"package_manifest.json"}:
        raise ValueError("Package has unexpected/missing files")
    return {"status": "passed", "files_verified": len(manifest["files_sha256"]), "weights_included": False, "secrets_env_included": False}


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle = (OUTPUT / NAME).resolve()
    if not bundle.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Bundle target escapes handoff output")
    if bundle.exists() and (not (bundle / "package_manifest.json").is_file() or load(bundle / "package_manifest.json").get("schema") != "m2.handoff.package.v1"):
        raise ValueError("Existing folder is not an owned package; refusing to replace it")
    if bundle.exists():
        check(bundle)  # Protect any manual edits to an earlier package.
    bundle.mkdir(exist_ok=True)
    files, annotations = selected_files()
    for relative, source in sorted(files.items()):
        target = (bundle / relative).resolve()
        if not target.is_relative_to(bundle) or not source.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Package copy path escapes the workspace or bundle")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "README.md", bundle / "README.md")
    for directory in ("output", "tmp"):
        (bundle / directory).mkdir(exist_ok=True)
        (bundle / directory / ".gitkeep").write_text("", encoding="utf-8")
    add_samples(bundle, annotations)
    hashes = {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "package_manifest.json"}
    write(bundle / "package_manifest.json", {"schema": "m2.handoff.package.v1", "date": "2026-11-14",
          "scope": "Shared M2 integration handoff excluding personal docs; product retrieval and group-data requirements remain incomplete",
          "weights_required_sha256": "d0d45d5299395d3a3ed0202596d3b22fa11494190a271c5b5adf429d6a9391fe",
          "files_sha256": hashes})
    result = check(bundle)
    archive = OUTPUT / (NAME + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                relative = path.relative_to(bundle).as_posix()
                info = zipfile.ZipInfo(NAME + "/" + relative, date_time=(2026, 11, 14, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                stream.writestr(info, path.read_bytes())
    result.update(archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive), bytes=archive.stat().st_size)
    write(OUTPUT / (NAME + ".check.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an extracted package instead of building")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.check)))
    else:
        build()
