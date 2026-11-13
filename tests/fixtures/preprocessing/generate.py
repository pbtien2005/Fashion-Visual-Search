"""Regenerate the synthetic handoff fixtures from the repository root."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from object_detection.image_preprocessing import PreprocessConfig, prepare_image


def main():
    folder = Path(__file__).resolve().parent
    metadata = {}
    for name, background, marker, orientation in (
        ("dark", (28, 35, 42), (64, 32, 16), 1),
        ("normal", (198, 199, 201), (230, 120, 100), 1),
        ("exif6", (198, 199, 201), (230, 30, 40), 6),
    ):
        image = Image.new("RGB", (96, 64), background)
        ImageDraw.Draw(image).rectangle((0, 0, 24, 20), fill=marker)
        if orientation == 1:
            upload = folder / f"{name}.png"
            image.save(upload)
        else:
            upload = folder / f"{name}.jpg"
            exif = image.getexif()
            exif[274] = orientation
            image.save(upload, exif=exif, quality=95)
        metadata[name] = {"upload": upload.name, "modes": {}}
        payload = upload.read_bytes()
        for mode in ("identity", "gamma"):
            prepared = prepare_image(payload, PreprocessConfig(mode=mode))
            outputs = {}
            for key in ("source_rgb", "detection_rgb"):
                output = folder / f"{name}.{mode}.{key}.png"
                Image.fromarray(getattr(prepared, key)).save(output)
                outputs[key] = output.name
            quality = dict(prepared.quality_meta)
            quality["preprocess_ms"] = None  # Timing is measured in the benchmark.
            metadata[name]["modes"][mode] = {
                **outputs,
                "schema_version": prepared.schema_version,
                "geometry_meta": prepared.geometry_meta,
                "quality_meta": quality,
            }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
