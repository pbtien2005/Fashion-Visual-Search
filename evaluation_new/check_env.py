"""Print package versions and whether the benchmark inputs are present."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# ultralytics (YOLO) and boto3 (R2) are not needed here: they run in the project Docker images.
PACKAGES = ("torch", "torchvision", "transformers", "open_clip_torch", "timm", "opencv-python-headless",
            "numpy", "pillow", "pandas", "matplotlib", "psycopg", "jupyterlab")
INPUTS = {
    "YOLO checkpoint": (Path("object_detection/best.pt"),
                        "d0d45d5299395d3a3ed0202596d3b22fa11494190a271c5b5adf429d6a9391fe"),
    "Catalog 1.752 sản phẩm": (Path("data/fashion_with_vectors.json"), None),
    "Manifest test": (Path("evaluation/bench_test.csv"), None),
    "Manifest validation": (Path("evaluation/bench_validation.csv"), None),
    "Nhãn qrels": (Path("docs/benchmark/main/bench_qrels.csv"), None),
}


def main():
    for name in PACKAGES:
        try:
            print(f"{name:18} {version(name)}")
        except PackageNotFoundError:
            print(f"{name:18} THIẾU")
    import torch

    print(f"{'torch threads':18} {torch.get_num_threads()}  cuda={torch.cuda.is_available()}")
    print()
    for label, (path, expected) in INPUTS.items():
        if not path.exists():
            print(f"THIẾU  {label}: {path}")
            continue
        status = "OK   "
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            status = "SAI HASH"
        print(f"{status} {label}: {path}")


if __name__ == "__main__":
    main()
