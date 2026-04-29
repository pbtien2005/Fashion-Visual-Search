import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.pipeline import run_etl_pipeline


KEYWORDS = [
    # Men's tops
    "men graphic oversized t-shirt vintage",
    "men plain v-neck slim fit tee",
    "men striped polo shirt cotton",
    "men long sleeve henley waffle knit",
    "men plaid flannel shirt lumberjack",
    "men floral hawaiian aloha shirt",
    "men oxford button down dress shirt",
    "men vintage wash denim jacket",
    "men lightweight bomber jacket",
    "men faux leather motorcycle jacket",
    "men zip up fleece hoodie",
    "men cable knit pullover sweater",
    # Men's bottoms
    "men ripped skinny jeans dark wash",
    "men straight leg relaxed fit light blue jeans",
    "men slim fit stretch chino pants khaki",
    "men tactical cargo pants multi pocket camo",
    "men athletic jogger sweatpants tapered",
    "men casual flat front chino shorts",
    "men quick dry board shorts tropical",
    # Women's tops
    "women floral chiffon blouse long sleeve",
    "women ribbed knit crop top sleeveless",
    "women peplum top sweetheart neckline",
    "women oversized graphic tee y2k style",
    "women elegant silk camisole cowl neck",
    "women oversized office blazer neutral",
    "women double breasted classic trench coat",
    "women chunky knit cardigan sweater oversized",
    "women cropped puffer jacket winter",
    "women fitted turtleneck ribbed sweater",
    # Women's bottoms and skirts
    "women high waisted wide leg palazzo pants",
    "women flared yoga pants crossover waist",
    "women vintage mom jeans high rise",
    "women faux leather skinny leggings",
    "women pleated tennis skirt plaid",
    "women satin midi slip skirt solid",
    "women distressed denim mini skirt",
    "women boho tiered maxi skirt floral",
    # Women's dresses
    "women summer floral wrap midi dress",
    "women ribbed knit bodycon mini dress",
    "women elegant satin slip maxi dress",
    "women casual short sleeve t-shirt dress",
    "women long sleeve sweater dress winter",
]


if __name__ == "__main__":
    max_items = int(os.getenv("ETL_MAX_ITEMS_PER_KEYWORD", "20"))
    keyword_start = int(os.getenv("ETL_KEYWORD_START", "0"))
    keyword_limit = int(os.getenv("ETL_KEYWORD_LIMIT", str(len(KEYWORDS))))

    selected_keywords = KEYWORDS[keyword_start : keyword_start + keyword_limit]
    print(
        f"Running {len(selected_keywords)} keywords "
        f"from index {keyword_start} with max_items={max_items}."
    )

    for keyword in selected_keywords:
        run_etl_pipeline(search_keyword=keyword, max_items=max_items)
