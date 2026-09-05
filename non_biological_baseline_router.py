from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,confusion_matrix,balanced_accuracy_score,f1_score
import external_physical_severity_high_router_experiment as base

OUT=Path("non_biological_baseline_router_results"); OUT.mkdir(exist_ok=True)
SAFE_CATS=[c for c in base.BASE_CAT if c!="eventScale"]

def met(y,p):
    rec=recall_score(y,p,labels=[3,4,5],average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=[3,4,5])
    return {
      "r3":float(rec[0]),"r4":float(rec[1]),"r5":float(rec[2]),
      "correct3":int(cm[0,0]),"correct4":int(cm[1,1]),"correct5":int(cm[2,2]),
      "n3":int((y==3).sum()),"n4":int((y==4).sum()),"n5":int((y==5).sum()),
      "min_recall":float(rec.min()),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "confusion_matrix":cm.tolist(),"pass80":bool(np.all(rec>=.8))
    }

def main():
    d=base.load_master()
    allh=d[d.band.isin([3,4,5])].copy()
    h=allh[allh.incidentType.astype(str).str.upper().str.strip()!="BIOLOGICAL"].copy().reset_index(drop=True)
    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in SAFE_CATS if c in h.columns]
    pred=np.full(len(h),-99,int)
    for yr in sorted(h.fyDeclared.astype(int).unique()):
        tr=h.fyDeclared.astype(int)!=yr; te=~tr
        Xtr=base.prep(h.loc[tr],nums,cats); Xte=base.prep(h.loc[te],nums,cats)
        m=base.fit_cat(Xtr,h.loc[tr,"band"].astype(int),Xte,cats)
        pred[te]=np.asarray(m.predict(Xte)).reshape(-1).astype(int)
    mm=met(h.band.astype(int).to_numpy(),pred)
    q=h[["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy(); q["predicted_band"]=pred
    q.to_csv(OUT/"predictions.csv",index=False)
    summary={
      "original_high_count":int(len(allh)),
      "removed_biological_count":int((allh.incidentType.astype(str).str.upper().str.strip()=="BIOLOGICAL").sum()),
      "remaining_count":int(len(h)),
      "remaining_band_counts":{str(b):int((h.band==b).sum()) for b in [3,4,5]},
      "remaining_incident_band_counts":h.groupby(["incidentType","band"]).size().reset_index(name="n").to_dict("records"),
      "metrics":mm,
      "guardrails":[
        "Biological removed from both training and evaluation.",
        "Strict leave-fiscal-year-out validation retained.",
        "Funding-derived eventScale excluded.",
        "No external features are used in this stripped-down confirmation."
      ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__": main()
