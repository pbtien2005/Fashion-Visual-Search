import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl.pipeline import run_etl_pipeline

if __name__ == "__main__":
    my_keywords = [
        # === ÁO NAM (Men's Tops) ===
        # "men graphic oversized t-shirt vintage",
        # "men plain v-neck slim fit tee",
        # "men striped polo shirt cotton",
        # "men long sleeve henley waffle knit",
        # "men plaid flannel shirt lumberjack",
        # "men floral hawaiian aloha shirt",
        # "men oxford button down dress shirt",
        # "men vintage wash denim jacket",
        # "men lightweight bomber jacket",
        # "men faux leather motorcycle jacket",
        # "men zip up fleece hoodie",
        # "men cable knit pullover sweater",
        
        # === QUẦN NAM (Men's Bottoms) ===
        # "men ripped skinny jeans dark wash",
        # "men straight leg relaxed fit light blue jeans",
        # "men slim fit stretch chino pants khaki",
        # "men tactical cargo pants multi pocket camo",
        # "men athletic jogger sweatpants tapered",
        # "men casual flat front chino shorts",
        # "men quick dry board shorts tropical",
        
        # === ÁO NỮ (Women's Tops) ===
        "women floral chiffon blouse long sleeve",
        "women ribbed knit crop top sleeveless",
        # "women peplum top sweetheart neckline",
        # "women oversized graphic tee y2k style",
        # "women elegant silk camisole cowl neck",
        # "women oversized office blazer neutral",
        # "women double breasted classic trench coat",
        # "women chunky knit cardigan sweater oversized",
        # "women cropped puffer jacket winter",
        # "women fitted turtleneck ribbed sweater",
        
        # === QUẦN & CHÂN VÁY NỮ (Women's Bottoms & Skirts) ===
        # "women high waisted wide leg palazzo pants",
        # "women flared yoga pants crossover waist",
        # "women vintage mom jeans high rise",
        # "women faux leather skinny leggings",
        # "women pleated tennis skirt plaid",
        # "women satin midi slip skirt solid",
        # "women distressed denim mini skirt",
        # "women boho tiered maxi skirt floral",
        
        # # === VÁY LIỀN NỮ (Women's Dresses) ===
        # "women summer floral wrap midi dress",
        # "women ribbed knit bodycon mini dress",
        # "women elegant satin slip maxi dress",
        # "women casual short sleeve t-shirt dress",
        # "women long sleeve sweater dress winter"
    ]
    for i in my_keywords:
        run_etl_pipeline(search_keyword=i, max_items=20)
