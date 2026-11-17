"""Resolve the existing clothing weights for both local and container runs."""

import os
from pathlib import Path


def resolve_model_path(model_path: str | None = None) -> str:
    configured_path = model_path or os.getenv("OBJECT_DETECTION_MODEL_PATH")
    if configured_path:
        # A bad explicit configuration must remain visible, rather than load another model.
        return str(configured_path)

    local_weight = Path(__file__).resolve().with_name("best.pt")
    candidates = (
        Path("/models/object-detection/best.pt"),
        local_weight,
        local_weight.parent.parent / "models" / "object-detection" / "best.pt",
    )
    return str(next((path for path in candidates if path.is_file()), local_weight))
