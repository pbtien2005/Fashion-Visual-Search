import os

# YOLO often confuses these three classes (blazer vs shirt-jacket vs gilet). When the flag is on,
# they share one keyword set so a mislabeled detection does not zero the category score of the
# right products. Default off: production behaviour is unchanged.
UPPER_LAYER_CLASSES = ("long_sleeve_top", "long_sleeve_outwear", "vest")
UPPER_LAYER_TERMS = ["top", "shirt", "blouse", "long sleeve", "jacket", "coat", "outwear",
                     "blazer", "cardigan", "shacket", "overshirt", "vest"]

YOLO_TO_DB_CATEGORY = {
    "short_sleeve_top": ["top", "shirt", "t-shirt", "tee"],
    "long_sleeve_top": ["top", "shirt", "blouse", "long sleeve"],
    "long_sleeve_outwear": ["jacket", "coat", "outwear"],
    "vest": ["vest"],
    "shorts": ["shorts"],
    "trousers": ["pants", "trousers", "jeans", "chino"],
    "skirt": ["skirt"],
    "short_sleeve_dress": ["dress"],
    "long_sleeve_dress": ["dress"],
    "vest_dress": ["dress"],
}

CLASS_NAME_VI = {
    "short_sleeve_top": "Áo tay ngắn",
    "long_sleeve_top": "Áo tay dài",
    "long_sleeve_outwear": "Áo khoác tay dài",
    "vest": "Áo vest",
    "shorts": "Quần short",
    "trousers": "Quần dài",
    "skirt": "Váy",
    "short_sleeve_dress": "Đầm tay ngắn",
    "long_sleeve_dress": "Đầm tay dài",
    "vest_dress": "Đầm vest",
}


def upper_layer_grouping_enabled() -> bool:
    return os.getenv("CATEGORY_GROUP_UPPER", "").strip().lower() in {"1", "true", "yes", "on"}


def get_category_terms(class_name: str) -> list[str]:
    if class_name in UPPER_LAYER_CLASSES and upper_layer_grouping_enabled():
        return list(UPPER_LAYER_TERMS)
    return YOLO_TO_DB_CATEGORY.get(class_name, [])


def get_class_name_vi(class_name: str) -> str:
    return CLASS_NAME_VI.get(class_name, class_name)
