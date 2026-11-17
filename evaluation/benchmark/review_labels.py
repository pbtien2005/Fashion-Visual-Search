"""Build a human review page for a sample of AI-graded product judgments.

The page shows, per sampled query, the query crop and every judged product that matters for the
metrics (the AI's acceptable products plus the top-5 of B0, C2 and C3_selected). The reviewer
sees the AI's flags, changes the ones they disagree with and exports a file in the exact format
that `grade merge` accepts, so reviewed pairs become human-confirmed (grader = reviewer name).
"""
import argparse
import json
import random
from pathlib import Path

from .catalog import CATALOG_PATH, Catalog
from .grade import CACHE, load_judgments, read_json

CONFIGS = ("B0_current", "C2_embedding_only", "C3_selected", "C3_w1")

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rà soát nhãn sản phẩm</title><style>
body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f4f5f3;color:#17261f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd;padding:10px 16px;z-index:5;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{padding:12px 16px;max-width:1400px;margin:auto}
.q{background:#fff;border:1px solid #cdd;border-radius:8px;margin:14px 0;padding:12px}
.q.done{border-color:#1b7f4b;background:#f1faf4}
.top{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.top img{max-height:260px;border-radius:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-top:10px}
.card{border:1px solid #cdd;border-radius:6px;padding:6px;background:#fafafa;font-size:13px}
.card.ai-acc{border:2px solid #1b7f4b}
.card.changed{background:#fff4d6;border-color:#c98a00}
.card img{width:100%;height:150px;object-fit:contain;background:#fff}
.card label{display:block;cursor:pointer}
.t{height:2.6em;overflow:hidden}
button{padding:6px 12px;border:1px solid #567;border-radius:6px;background:#fff;cursor:pointer}
button.p{background:#145b46;color:#fff;border-color:#145b46}
small{color:#556}
</style></head><body>
<header><b>Rà soát nhãn sản phẩm</b>
<label>Tên người rà <input id="name" placeholder="họ tên"></label>
<span id="count"></span><button class="p" id="export">Xuất file</button></header>
<main><p>Mỗi thẻ hiện nhãn AI đã chấm. <b>Chỉ bỏ hoặc thêm dấu tick khi bạn không đồng ý.</b> Viền xanh = AI chấp nhận.
Ô vàng = bạn đã sửa. Tiêu chí: <b>Đúng loại</b> theo món nhìn thấy trong khung xanh; <b>Đúng màu</b> khớp màu chủ đạo ghi ở tiêu đề; <b>Chấp nhận</b> = đúng loại, đúng màu, và kiểu dáng đủ gần.
Xong một ảnh thì bấm "Đã rà xong ảnh này"; chỉ ảnh đã rà xong mới được xuất.</p><div id="app"></div></main>
<script>
const DATA=__DATA__;const KEY="review_labels_v1";let saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const st=saved.st||{};const done=saved.done||{};const nameEl=document.getElementById("name");nameEl.value=saved.name||"";
function persist(){try{localStorage.setItem(KEY,JSON.stringify({st,done,name:nameEl.value}))}catch(e){}}
nameEl.oninput=persist;
function cur(q,p){return st[q.id+"|"+p.id]||{type_ok:p.type_ok,color_ok:p.color_ok,acceptable:p.acceptable}}
function render(){const app=document.getElementById("app");app.innerHTML="";let n=0;
DATA.queries.forEach(q=>{const d=!!done[q.id];n+=d;const el=document.createElement("section");el.className="q"+(d?" done":"");
el.innerHTML=`<div class="top"><img src="${q.image}"><div><h3>${q.id} <small>${q.stage}</small></h3><div>Loại nhãn: <b>${q.class_name}</b> · Màu nhãn: <b>${q.color}</b></div>
<small>${q.cards.length} sản phẩm</small><br><button data-q="${q.id}">${d?"Bỏ đánh dấu":"Đã rà xong ảnh này"}</button></div></div><div class="cards"></div>`;
const box=el.querySelector(".cards");q.cards.forEach(p=>{const s=cur(q,p);const ch=s.type_ok!==p.type_ok||s.color_ok!==p.color_ok||s.acceptable!==p.acceptable;
const c=document.createElement("div");c.className="card"+(p.acceptable?" ai-acc":"")+(ch?" changed":"");
c.innerHTML=`<a href="${p.url}" target="_blank"><img src="${p.image}" loading="lazy"></a><div class="t">${p.title}</div>
<label><input type="checkbox" data-k="type_ok" ${s.type_ok?"checked":""}> Đúng loại</label>
<label><input type="checkbox" data-k="color_ok" ${s.color_ok?"checked":""}> Đúng màu</label>
<label><input type="checkbox" data-k="acceptable" ${s.acceptable?"checked":""}> <b>Chấp nhận</b></label>`;
c.querySelectorAll("input").forEach(i=>i.onchange=()=>{const x={...cur(q,p)};x[i.dataset.k]=i.checked?1:0;
if(x.acceptable&&(!x.type_ok||!x.color_ok)){x.type_ok=1;x.color_ok=1}if(!x.type_ok)x.acceptable=0;st[q.id+"|"+p.id]=x;persist();render()});
box.appendChild(c)});
el.querySelector("button").onclick=()=>{done[q.id]=!done[q.id];persist();render()};app.appendChild(el)});
document.getElementById("count").textContent=n+"/"+DATA.queries.length+" ảnh đã rà xong"}
document.getElementById("export").onclick=()=>{const name=nameEl.value.trim();if(!name){alert("Nhập tên người rà.");nameEl.focus();return}
const pairs=[],queries=[];DATA.queries.forEach(q=>{if(!done[q.id])return;queries.push({query_id:q.id,no_match:false,expanded_reviewed:false});
q.cards.forEach(p=>{const s=cur(q,p);pairs.push({query_id:q.id,product_id:p.id,type_ok:s.type_ok,color_ok:s.color_ok,acceptable:s.acceptable})})});
if(!queries.length){alert("Chưa có ảnh nào được đánh dấu đã rà xong.");return}
const changed=pairs.filter(p=>{const o=DATA.queries.find(q=>q.id===p.query_id).cards.find(c=>c.id===p.product_id);return o.type_ok!==p.type_ok||o.color_ok!==p.color_ok||o.acceptable!==p.acceptable}).length;
const blob=new Blob([JSON.stringify({grader:name,exported_at:new Date().toISOString(),pairs,queries},null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="grades_review_human.json";a.click();
alert("Đã xuất "+queries.length+" ảnh, "+pairs.length+" cặp, trong đó bạn sửa "+changed+" cặp.")};
render();
</script></body></html>"""


def top_ids(result, count):
    index = result["target"]["index"]
    if index is None:
        return []
    return [product["id"] for product in result["detections"][index]["ranked"][:count]]


def build(run_dir, stages, per_stage, seed):
    pairs, statuses = load_judgments()
    catalog = Catalog.load(CATALOG_PATH)
    rng = random.Random(seed)
    queries = []
    for stage in stages:
        snapshot = read_json(run_dir / stage / "results.json")
        eligible = [query for query in snapshot["queries"] if statuses.get(query["query_id"]) == "has_match"]
        for query in rng.sample(eligible, min(per_stage[stage], len(eligible))):
            qid = query["query_id"]
            ids = {pid for (judged, pid), pair in pairs.items() if judged == qid and pair["acceptable"]}
            for config in CONFIGS:
                result = query["results"].get(config)
                if result:
                    ids.update(top_ids(result, 5))
            cards = []
            for pid in sorted(ids):
                pair = pairs.get((qid, pid))
                product = catalog.get(pid)
                if pair is None or product is None:
                    continue
                cards.append({"id": pid, "title": (product.get("title") or pid)[:90],
                              "url": product.get("product_url") or f"https://www.amazon.com/dp/{pid}",
                              "image": f"../../../benchmark/cache/products/{pid}.jpg",
                              "type_ok": int(pair["type_ok"]), "color_ok": int(pair["color_ok"]),
                              "acceptable": int(pair["acceptable"])})
            queries.append({"id": qid, "stage": stage, "class_name": query["class_name"], "color": query["color"],
                            "image": f"../../../benchmark/cache/queries/{qid}.jpg", "cards": cards})
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--validation", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    queries = build(args.run_dir, ("test", "validation"), {"test": args.test, "validation": args.validation}, args.seed)
    out = args.run_dir / "review"
    out.mkdir(exist_ok=True)
    payload = json.dumps({"queries": queries}, ensure_ascii=False).replace("<", "\\u003c")
    (out / "review.html").write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    print(out / "review.html", len(queries), "queries,", sum(len(q["cards"]) for q in queries), "cards")
    assert CACHE.exists()


if __name__ == "__main__":
    main()
