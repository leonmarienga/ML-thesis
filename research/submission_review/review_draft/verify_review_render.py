"""Verify manuscript links, exact referenced sources and package structure; no data access."""
import argparse, hashlib, json, re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

parser=argparse.ArgumentParser()
for x in ("docx","layout_manifest","source_tree","followup_tree","source"):
    parser.add_argument(x,type=Path)
parser.add_argument("source_ref")
p=parser.parse_args()
digest=lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
layout=json.loads(p.layout_manifest.read_bytes())
assert layout["docx_sha256"]==digest(p.docx)
assert layout["manuscript_sha256"]==digest(p.source)
assert layout["office_display_math"]==19 and len(layout["figures"])==11
w="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
root="https://github.com/leonmarienga/ML-thesis/blob/"
followup="6bd74e7dceaf339df2b9f911141a8a7281428d5f"
wanted={
 "research/followup_20261128/target_blind_router_v1/evidence/summary.json":"0b150bf56af4ac995f6863d246c8d37b0177c1f7",
 "research/followup_20261128/uncertainty_audit/REPORT.md":"b7835e1c8d0156be4636c0690fd5eccf56ac1368",
 "research/followup_20261128/policy_veto_fixture/REPORT.md":"e2878ae7a6499986406cc90772f40cbd40ed3788",
}
def trees(path):
    obj=json.loads(path.read_bytes())
    assert obj["truncated"] is False
    return {x["path"]:x["sha"] for x in obj["tree"] if x["type"]=="blob"}
pinned=trees(p.source_tree); follow=trees(p.followup_tree)
assert all(follow.get(k)==v for k,v in wanted.items())
assert pinned.get("research/submission_review/THESIS.md")==hashlib.sha1(
    f"blob {p.source.stat().st_size}\0".encode()+p.source.read_bytes()).hexdigest()
with ZipFile(p.docx) as z:
    assert z.testzip() is None
    doc=ET.fromstring(z.read("word/document.xml"))
    names={u.get(w+"name") for u in doc.iter(w+"bookmarkStart")}
    anchors={u.get(w+"anchor") for u in doc.iter(w+"hyperlink") if u.get(w+"anchor")}
    assert anchors<=names, sorted(anchors-names)
    rs=ET.fromstring(z.read("word/_rels/document.xml.rels"))
    links={u.get("Target") for u in rs if u.get("TargetMode")=="External"}
    assert not any("/blob/thesis/" in u or "/tree/thesis/" in u for u in links)
    direct={root+followup+"/"+k for k in wanted}
    assert direct<=links, sorted(direct-links)
    proof={u for u in links if u.startswith(root+p.source_ref+"/")}
    relative=set(re.findall(r"(?<!!)\[[^\n\]]+\]\(([^\s):]+(?:/[^\s)]+|\.md))\)",p.source.read_text()))
    expected={root+p.source_ref+"/research/submission_review/"+q for q in relative if not q.startswith(("#","/"))}
    assert len(expected)==15 and expected==proof, (expected-proof,proof-expected)
    assert all(u.split("#",1)[0][len(root+p.source_ref)+1:] in pinned for u in proof)
    print(json.dumps({"internal_bookmark_destinations":len(anchors),"source_relative_pins":len(proof),
      "exact_followup_citations":len(direct),"figures":len(layout["figures"]),
      "native_display_equations":layout["office_display_math"]},sort_keys=True))
