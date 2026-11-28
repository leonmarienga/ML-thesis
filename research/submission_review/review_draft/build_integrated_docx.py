"""Build a styled supervisor-review draft from the integrated Markdown, not a final EUL template."""
import argparse, hashlib, json, os, re, shutil, subprocess, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

p=argparse.ArgumentParser()
for arg in ("source","transport_figure","output"):
    p.add_argument(arg,type=Path)
p.add_argument("--review-ref",required=True)
p.add_argument("--figure-manifest",type=Path, required=True)
a=p.parse_args()
assert len({x.resolve() for x in (a.source,a.transport_figure,a.output,a.figure_manifest)})==4, "Input and output paths must differ"
branchurl="https://github.com/leonmarienga/ML-thesis/blob/"+a.review_ref+"/research/submission_review/"
source=a.source.read_text(encoding="utf-8")
title,body=source.split("\n",1)
assert title.startswith("# ") and all(f"## Chapter {i} " in body for i in range(1,7))
heading=title[2:]
body=re.sub(r"^Leon Marienga \|.*$","",body,count=1,flags=re.M)
# Stage each actual source figure by identity before Pandoc; there is no fallback to a missing SVG.
pattern=r"!\[([^\n\]]+)\]\((figures/(?:prototype/)?(F(?:10|[1-9]))-[A-Za-z0-9_.-]+\.(?:svg|png))\)"
image_matches=list(re.finditer(pattern,body))
figures={m[3]:a.source.resolve().parent/m[2] for m in image_matches}
assert len(figures)==len(image_matches) and set(figures)=={f"F{i}" for i in range(1,11)}
assert body.count("[FIGURE:transport]")==1
def stage_markup(m):
    description=m[1] if m[1].startswith(f"Figure {m[3]}.") else f"Figure {m[3]}. {m[1]}"
    return f"![{description}]({m[3]}.png){{width=6.1in}}"
body=re.sub(pattern,stage_markup,body)
body=body.replace("[FIGURE:transport]","![Figure B1. Tested transport allocation frontier](B1.png){width=6.1in}")
body=re.sub(r"(?m)^(##) ",r"# ",body)
body=re.sub(r"(?m)^(###) ",r"## ",body)
body=re.sub(r"(?<!\!)\[([^\n\]]+)\]\(([^\s):]+(?:/[^\s)]+|\.md))\)",
            lambda m:f"[{m[1]}]({branchurl}{m[2]})" if not m[2].startswith(("#","/")) else m[0],body)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory() as td:
    w=Path(td)
    tracked={}
    for tag,src in figures.items():
        png=w/f"{tag}.png"
        if src.suffix==".png":
            shutil.copyfile(src,png)
        else:
            subprocess.run(["inkscape",str(src),"--export-width=2000","--export-filename="+str(png)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        tracked[tag]={"source":src.name,"source_sha256":sha(src),"png_sha256":sha(png)}
    shutil.copyfile(a.transport_figure,w/"B1.png")
    tracked["B1"]={"source":a.transport_figure.name,"source_sha256":sha(a.transport_figure),"png_sha256":sha(w/"B1.png")}
    (w/"manuscript.md").write_text(body,encoding="utf-8")
    out=w/"raw.docx"
    subprocess.run(["pandoc","-f","markdown+pipe_tables-implicit_figures+tex_math_single_backslash-tex_math_dollars","-t","docx","-o",str(out),"manuscript.md"],cwd=w,check=True)
    doc=Document(out)
    normal=doc.styles["Normal"]
    normal.font.name="Cambria"
    normal.font.size=Pt(11.5)
    normal.paragraph_format.line_spacing=1.14
    normal.paragraph_format.space_after=Pt(6)
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
    for content,style in [(heading,"Title"),("Leon Marienga","Subtitle"),("MSc Computer Engineering","Subtitle"),("European University of Lefke","Subtitle"),("Supervisor Review Draft | 28 November 2026","Subtitle")]:
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
        if pa.style.name=="Heading 1" and (pa.text.startswith("Chapter 1 ") or pa.text in {"List of figures"}):
            pa.paragraph_format.page_break_before=True
        if re.match(r"^Table T(?:1[01]|[1-9])\.",pa.text):
            pa.paragraph_format.keep_with_next=True
            pa.paragraph_format.keep_together=True
        if pa._p.xpath(".//w:drawing"):
            pa.paragraph_format.keep_with_next=True
            pa.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif re.match(r"^Figure (?:F(?:10|[1-9])|B1)\.",pa.text):
            pa.style=doc.styles["Caption"]
            pa.paragraph_format.keep_together=True
            pa.paragraph_format.widow_control=True
    for table in doc.tables:
        hdr=table.rows[0]._tr.get_or_add_trPr()
        hdr.append(OxmlElement("w:tblHeader"))
        widths={4:(1.92,.62,1.92,1.82),7:(.62,1.48,.78,.83,.58,.94,1.05)}.get(len(table.columns))
        first=table.rows[0].cells[0].text
        if first=="Source or feature":
            widths=(1.76,1.15,.73,2.64)
        elif first=="Analysis":
            widths=(1.15,.91,.58,1.28,2.36)
        elif first in ("Version and status","Evidence","Study","Component"):
            widths=(1.48,1.82,2.98) if first!="Study" else (1.64,1.32,3.32)
        elif first=="Nominal":
            widths=(.80,1.44,.77,.83,.59,.89,.96)
        elif first.startswith("Marker and"):
            widths=(1.27,.91,1.19,.61,1.04,1.26)
        if widths:
            table.autofit=False
            for col,width in zip(table.columns,widths):
                col.width=Inches(width)
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            if widths:
                for cell,width in zip(row.cells,widths):
                    cell.width=Inches(width)
                    for par in cell.paragraphs:
                        par.paragraph_format.line_spacing=1.03
                        par.paragraph_format.space_before=Pt(0)
                        par.paragraph_format.space_after=Pt(3)
                        for run in par.runs:
                            run.font.size=Pt(10)
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
    candidate=w/"verified-candidate.docx"
    doc.save(candidate)
    with ZipFile(candidate) as z:
        assert z.testzip() is None
        ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "a":"http://schemas.openxmlformats.org/drawingml/2006/main",
            "wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
            "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
            "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        xml=z.read("word/document.xml").decode()
        assert "FIGURE:transport" not in xml and all(f"Chapter {i}" in xml for i in range(1,7))
        assert all(s in xml for s in ("List of figures","List of tables","MSc Computer Engineering","Keywords:"))
        assert "$5,014,819.01" in xml and "Prospective tabletop evaluation" in xml and "$2.155B" in xml
        tree=ET.fromstring(xml)
        rels={v.attrib["Id"]:v.attrib for v in ET.fromstring(z.read("word/_rels/document.xml.rels"))}
        pars=list(tree.findall("./w:body/w:p",ns))
        mapped={}
        for j,par in enumerate(pars):
            graphic=par.find(".//a:blip",ns)
            if graphic is None:
                continue
            prop=par.find(".//wp:docPr",ns)
            assert prop is not None
            alt=prop.attrib.get("descr","") or prop.attrib.get("title","")
            mat=re.match(r"Figure (F(?:10|[1-9])|B1)\.",alt)
            assert mat, ("Missing figure alt identity",prop.attrib)
            tag=mat[1]
            text_next="".join(t.text or "" for t in pars[j+1].findall(".//w:t",ns))
            assert text_next.startswith(f"Figure {tag}."),(tag,text_next)
            assert tag not in mapped
            r_id=graphic.attrib["{"+ns["r"]+"}embed"]
            relation=rels[r_id]
            media=relation["Target"]
            assert media.startswith("media/") and "/" not in media[len("media/"):]
            assert relation["Type"]==ns["r"]+"/image"
            data=z.read("word/"+media)
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            assert hashlib.sha256(data).hexdigest()==tracked[tag]["png_sha256"]
            mapped[tag]=media
        package_media={n[len("word/"):] for n in z.namelist() if n.startswith("word/media/")}
        assert mapped.keys()==tracked.keys() and set(mapped.values())==package_media
        for tag,media in mapped.items():tracked[tag]["office_media_part"]=media
        math_count=len(tree.findall(".//m:oMathPara",ns))
        assert math_count>=19, f"Expected Appendix D display math, got {math_count}"
    manifest=w/"verified-manifest.json"
    manifest.write_text(json.dumps({"manuscript_sha256":sha(a.source),"docx_sha256":sha(candidate),
                                    "office_display_math":math_count,"figures":tracked},indent=2,sort_keys=True)+"\n")
    def promote(src,dest):
        # Use a sibling temp file so replacement is atomic per file on the destination filesystem.
        dest.parent.mkdir(parents=True,exist_ok=True)
        fd, temp_name=tempfile.mkstemp(prefix=".thesis-build-",dir=dest.parent)
        os.close(fd)
        part=Path(temp_name)
        try:
            shutil.copyfile(src,part)
            assert sha(src)==sha(part)
            os.replace(part,dest)
        finally:
            part.unlink(missing_ok=True)
    promote(candidate,a.output)
    promote(manifest,a.figure_manifest)
