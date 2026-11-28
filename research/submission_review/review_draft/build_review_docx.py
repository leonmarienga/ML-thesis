"""Build a styled supervisor-review draft from the integrated Markdown, not a final EUL template."""
import argparse, re, shutil, subprocess, tempfile
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

p=argparse.ArgumentParser()
for arg in ("source","historical_figure","transport_figure","output"):
    p.add_argument(arg,type=Path)
p.add_argument("--review-ref",required=True)
p.add_argument("--uncertainty-figure",type=Path)
a=p.parse_args()
branchurl="https://github.com/leonmarienga/ML-thesis/blob/"+a.review_ref+"/research/submission_review/"
source=a.source.read_text(encoding="utf-8")
title,body=source.split("\n",1)
assert title.startswith("# ") and all(f"## Chapter {i} " in body for i in range(1,7))
heading=title[2:]
body=re.sub(r"^Leon Marienga \|.*$","",body,count=1,flags=re.M)
# Retain exactly the published captions while embedding PNGs for Word interoperability.
assert "(figures/F6-track-a.svg)" in body and "[FIGURE:transport]" in body
body=body.replace("(figures/F6-track-a.svg)","(f6.png){width=6.1in}")
body=body.replace("[FIGURE:transport]","![Tested transport allocation frontier](allocation.png){width=6.1in}")
body=re.sub(r"(?m)^(##) ",r"# ",body)
body=re.sub(r"(?m)^(###) ",r"## ",body)
body=re.sub(r"(?<!\!)\[([^\n\]]+)\]\(([^\s):]+(?:/[^\s)]+|\.md))\)",
            lambda m:f"[{m[1]}]({branchurl}{m[2]})" if not m[2].startswith(("#","/")) else m[0],body)
has_f8="(figures/F8-matched-intervals.svg)" in body
if has_f8:
    assert a.uncertainty_figure is not None
    body=body.replace("(figures/F8-matched-intervals.svg)","(f8.png){width=6.1in}")
with tempfile.TemporaryDirectory() as td:
    w=Path(td)
    shutil.copyfile(a.historical_figure,w/"f6.png")
    shutil.copyfile(a.transport_figure,w/"allocation.png")
    if has_f8:
        shutil.copyfile(a.uncertainty_figure,w/"f8.png")
    (w/"manuscript.md").write_text(body,encoding="utf-8")
    out=w/"raw.docx"
    subprocess.run(["pandoc","-f","markdown+pipe_tables-implicit_figures-tex_math_dollars","-t","docx","-o",str(out),"manuscript.md"],cwd=w,check=True)
    doc=Document(out)
    normal=doc.styles["Normal"]
    normal.font.name="Cambria"
    normal.font.size=Pt(11.5)
    normal.paragraph_format.line_spacing=1.18
    normal.paragraph_format.space_after=Pt(8)
    for name,size in [("Title",27),("Subtitle",12),("Heading 1",19),("Heading 2",13),("Caption",10),("Footer",9)]:
        if name in doc.styles:
            st=doc.styles[name]
            st.font.name="Cambria"
            st.font.size=Pt(size)
            st.font.color.rgb=RGBColor(0,0,0)
            if name.startswith("Heading"):
                st.paragraph_format.keep_with_next=True
    if "Footer" not in [st.name for st in doc.styles]:
        fs=doc.styles.add_style("Footer",WD_STYLE_TYPE.PARAGRAPH)
        fs.font.name="Cambria";fs.font.size=Pt(9)
    sub=doc.paragraphs[0]
    for content,style in [(heading,"Title"),("Leon Marienga","Subtitle"),("Master’s Thesis Supervisor Review Draft","Subtitle")]:
        par=sub.insert_paragraph_before(content,style)
        par.alignment=WD_ALIGN_PARAGRAPH.CENTER
        if style=="Title":
            par.paragraph_format.space_before=Pt(95)
            par.paragraph_format.space_after=Pt(20)
    par=sub.insert_paragraph_before("Research draft for supervisor review. Institutional formatting, author disclosure, and approvals are subject to confirmation.")
    par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before=Pt(28)
    sub.paragraph_format.page_break_before=True
    for pa in doc.paragraphs:
        if pa.style.name in {"Title","Subtitle"} or pa.style.name.startswith("Heading"):
            for run in pa.runs:
                run.font.color.rgb=RGBColor(0,0,0)
        if pa.style.name=="Heading 1" and (pa.text.startswith("Chapter 1 ") or pa.text in {"Appendix A Reproduction and claim tracing"}):
            pa.paragraph_format.page_break_before=True
        if pa._p.xpath(".//w:drawing"):
            pa.paragraph_format.keep_with_next=True
            pa.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif pa.text.startswith(("Figure F6.","Figure F8.","Figure B1.")):
            pa.style=doc.styles["Caption"]
            pa.paragraph_format.keep_together=True
            pa.paragraph_format.widow_control=True
    for table in doc.tables:
        hdr=table.rows[0]._tr.get_or_add_trPr()
        hdr.append(OxmlElement("w:tblHeader"))
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    s=doc.sections[0]
    s.page_width=Inches(8.27);s.page_height=Inches(11.69)
    s.top_margin=s.bottom_margin=Inches(.85)
    s.left_margin=s.right_margin=Inches(.95)
    s.different_first_page_header_footer=True
    f=s.footer.paragraphs[0]
    f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    f.style=doc.styles["Footer"]
    field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");f._p.append(field)
    doc.core_properties.author="Leon Marienga"
    doc.core_properties.title=heading
    doc.core_properties.subject="Supervisor-review draft"
    doc.core_properties.comments="Derived from the integrated Markdown and existing research figures."
    a.output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(a.output)
    with ZipFile(a.output) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("word/media/")])==2+int(has_f8)
        xml=z.read("word/document.xml").decode()
        assert "FIGURE:transport" not in xml and all(f"Chapter {i}" in xml for i in range(1,7))
        assert "$5,014,819.01" in xml and "Prospective tabletop evaluation" in xml
        if has_f8:
            assert "$2.155B" in xml and "7/9" in xml
