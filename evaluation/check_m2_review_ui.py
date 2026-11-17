"""Build an isolated synthetic browser smoke page; it never edits real annotations.

Open the generated file with an existing Chromium browser in headless mode,
then inspect #qa-result from --dump-dom. No network or third-party scripts.
"""

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
<script>
const qaChecks=[];
function assertQA(value,name){if(!value)throw Error(name);qaChecks.push(name)}
async function runQA(){
 try{
  if(!ready){setTimeout(runQA,50);return}
  assertQA(canvas.width===128&&canvas.height===96,'canvas_geometry');
  el('bboxReviewed').click();assertQA(!el('bboxReviewed').checked,'review_requires_name');
  el('reviewer').value='SYNTHETIC_UI_TEST_ONLY';el('bboxReviewed').click();
  assertQA(current().label_status==='human_reviewed','explicit_target_review');
  el('tool').value='polygon';
  el('editor').open=true;
  canvas.setPointerCapture=()=>{}; // Synthetic events have no active hardware pointer.
  const rect=canvas.getBoundingClientRect(),scale=Math.min(rect.width/canvas.width,rect.height/canvas.height),w=canvas.width*scale,h=canvas.height*scale;
  for(const p of [[20,20],[90,20],[90,70],[20,70]])canvas.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,clientX:rect.left+(rect.width-w)/2+p[0]*scale,clientY:rect.top+(rect.height-h)/2+p[1]*scale}));
  el('close').click();assertQA(current().mask_polygons.length===1,'polygon_closed');
  assertQA(Math.abs(current().mask_polygons[0].points[0][0]-20)<.01,'display_to_working_coordinates');
  el('maskReviewed').click();assertQA(current().mask_status==='human_reviewed','explicit_mask_review');
  let capturedBlob=null,downloadName=null;URL.createObjectURL=blob=>{capturedBlob=blob;return 'blob:synthetic-test'};URL.revokeObjectURL=()=>{};
  HTMLAnchorElement.prototype.click=function(){downloadName=this.download};
  el('download').click();const exported=JSON.parse(await capturedBlob.text());
  assertQA(downloadName==='m2-review.json'&&exported.queries[0].mask_polygons.length===1,'single_JSON_export_contains_polygons');
  el('clear').click();assertQA(!current().mask_polygons.length&&current().mask_status==='pending_human_review','editing_invalidates_mask_review');
  el('tool').value='bbox';
  const client=p=>({bubbles:true,pointerId:1,clientX:rect.left+(rect.width-w)/2+p[0]*scale,clientY:rect.top+(rect.height-h)/2+p[1]*scale});
  canvas.dispatchEvent(new PointerEvent('pointerdown',client([10,10])));
  canvas.dispatchEvent(new PointerEvent('pointerup',client([100,80])));
  assertQA(Math.abs(current().bbox_normalized[0]-10/128)<.0001,'bbox_drag_uses_normalized_working_coordinates');
  assertQA(current().label_status==='pending_human_review','bbox_edit_invalidates_review');
  el('next').click();assertQA(current().query_id==='synthetic_ui_2'&&current().label_status==='pending_human_review','navigation_does_not_share_review_state');
  const files=new DataTransfer();files.items.add(new File([JSON.stringify(exported)],'resume.json',{type:'application/json'}));el('resume').files=files.files;el('resume').dispatchEvent(new Event('change'));
  await new Promise(resolve=>setTimeout(resolve,200));
  assertQA(state.queries[0].mask_polygons.length===1&&state.queries[0].mask_status==='human_reviewed','resume_restores_saved_work');
  window.qaResult={status:'passed',checks:qaChecks,scope:'synthetic_UI_fixture_only_no_human_labels_created'};
 }catch(error){window.qaResult={status:'failed',checks:qaChecks,error:String(error)}}
 const result=document.createElement('pre');result.id='qa-result';result.textContent=JSON.stringify(window.qaResult);document.body.prepend(result);
}
setTimeout(runQA,100);
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/m2-review-smoke.html")
    args = parser.parse_args()
    rows = []
    for index in (1, 2):
        rows.append(dict(query_id=f"synthetic_ui_{index}", target_id=f"synthetic_ui_{index}_target",
                         class_name="short_sleeve_top", bbox_normalized=[.1, .1, .9, .9],
                         split="validation", color_names=["blue"], lighting_group="normal",
                         geometry_meta=dict(width=128, height=96), source_url="#", sha256="synthetic_fixture",
                         label_status="pending_human_review", mask_status="not_annotated", mask_requested=True,
                         reviewer=None, mask_reviewer=None, previously_used_for_m2_development=False))
    picture = "data:image/png;base64," + base64.b64encode((ROOT / "tests/fixtures/m2/source_rgb.png").read_bytes()).decode()
    payload = dict(manifest=dict(schema_version="m2.annotation.v2", preprocess_config=dict(mode="identity"), queries=rows),
                   previews={row["query_id"]: picture for row in rows})
    template = (ROOT / "evaluation/m2_review.html").read_text(encoding="utf-8")
    page = template.replace("__M2_REVIEW_PAYLOAD__", json.dumps(payload)).replace("</html>", HARNESS + "</html>")
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
