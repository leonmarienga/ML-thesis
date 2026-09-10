#!/usr/bin/env python3
"""Deterministic integration of accepted 814 + strict Hurricane current19 RF rescue."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"hurricane_rf_integration"
OUT=ROOT/"audit_outputs"/"hurricane_rf_prediction_integration"
OUT.mkdir(parents=True,exist_ok=True)
ACCEPTED=IN/"accepted"/"integrated_predictions.csv"
HURR=IN/"hurricane"/"hurr_current_rf_predictions.csv"
BANDS=["0-100K","100K-1M","1M-50M","50-200M","200-500M","500M+"]

def metrics(df,col):
    per={}; rec=[]; correct=0
    for b in BANDS:
        m=df.actual_band==b; n=int(m.sum()); c=int((df.loc[m,col]==b).sum()); r=c/n if n else None
        per[b]={"correct":c,"total":n,"recall":r}; correct+=c
        if r is not None: rec.append(r)
    return {"correct":correct,"total":len(df),"accuracy":correct/len(df),"macro":sum(rec)/len(rec),"per_band":per}

def main():
    a=pd.read_csv(ACCEPTED); h=pd.read_csv(HURR)
    a.disasterNumber=a.disasterNumber.astype(int); h.disasterNumber=h.disasterNumber.astype(int)
    base_col="storm_integrated_pred"
    assert base_col in a.columns
    changed=h[h.final_pred!=h.base_pred].copy()
    assert len(changed)>0
    assert set(changed.incidentType)=={"Hurricane"}
    assert ((changed.base_pred=="100K-1M")&(changed.final_pred=="1M-50M")).all()
    m=changed[["disasterNumber","incidentType","actual_band","base_pred","final_pred"]].merge(
        a[["disasterNumber","actual_band",base_col,"root_actual_high","root_pred_high"]],
        on="disasterNumber",how="left",suffixes=("_hurr","_accepted"),validate="one_to_one")
    assert m[base_col].notna().all()
    assert (m[base_col]==m.base_pred).all()
    assert (m.root_actual_high==0).all() and (m.root_pred_high==0).all()
    out=a.copy(); out["hurricane_integrated_pred"]=out[base_col]
    rmap=dict(zip(changed.disasterNumber,changed.final_pred))
    z=out.disasterNumber.isin(rmap)
    out.loc[z,"hurricane_integrated_pred"]=out.loc[z,"disasterNumber"].map(rmap)
    high=out.root_actual_high==1
    assert (out.loc[high,"hurricane_integrated_pred"]==out.loc[high,base_col]).all()
    base=metrics(out,base_col); new=metrics(out,"hurricane_integrated_pred")
    m["base_correct"]=m.base_pred==m.actual_band_hurr
    m["hurricane_correct"]=m.final_pred==m.actual_band_hurr
    out.to_csv(OUT/"integrated_predictions.csv",index=False)
    m.to_csv(OUT/"hurricane_promotions.csv",index=False)
    summary={"source_runs":{"accepted_814":34442966733,"hurricane_rf":34443201744},"promotion_count":len(m),"promotion_disaster_numbers":m.disasterNumber.astype(int).tolist(),"base_metrics":base,"integrated_metrics":new,"net_correct_gain":new["correct"]-base["correct"],"macro_gain":new["macro"]-base["macro"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    md=["# Accepted 814 + Hurricane RF prediction integration","","- Completed strict artifacts only; no retraining or retuning.","- Every Hurricane promotion matches the accepted 814 low prediction.","- No actual or routed >=$50M row is modified.","","| Variant | Overall | Macro | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,q in [("accepted_814",base),("plus_hurricane_current_rf",new)]:
        p=q["per_band"]
        md.append(f"| {name} | {q['correct']}/{q['total']} ({q['accuracy']:.1%}) | {q['macro']:.1%} | {p['0-100K']['correct']}/{p['0-100K']['total']} ({p['0-100K']['recall']:.1%}) | {p['100K-1M']['correct']}/{p['100K-1M']['total']} ({p['100K-1M']['recall']:.1%}) | {p['1M-50M']['correct']}/{p['1M-50M']['total']} ({p['1M-50M']['recall']:.1%}) | {p['50-200M']['correct']}/{p['50-200M']['total']} ({p['50-200M']['recall']:.1%}) | {p['200-500M']['correct']}/{p['200-500M']['total']} ({p['200-500M']['recall']:.1%}) | {p['500M+']['correct']}/{p['500M+']['total']} ({p['500M+']['recall']:.1%}) |")
    md += ["","## Hurricane RF promotions","","| Disaster | Actual | Base | Specialist | Base correct | Specialist correct |","|---:|---|---|---|---:|---:|"]
    for _,r in m.iterrows(): md.append(f"| {int(r.disasterNumber)} | {r.actual_band_hurr} | {r.base_pred} | {r.final_pred} | {bool(r.base_correct)} | {bool(r.hurricane_correct)} |")
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))
if __name__=="__main__": main()
