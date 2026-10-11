"""Routed-to-$50M-$200M outer-LFYO check; unchanged 830/912 router."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nonbio_50_200m_fast_hybrid import load, select, candidate_preds, LO, HI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "nonbio_50_200m_routed_audit"

def metrics(y, p):
    y=np.asarray(y,float)*1e6; p=np.asarray(p,float)*1e6
    err=np.abs(y-p); relative=err/np.maximum(y,1.0)
    return {"n":len(y),"MAE_USD":float(err.mean()),
      "RMSE_USD":float(np.sqrt(np.mean((y-p)**2))),
      "MedAE_USD":float(np.median(err)),
      "R2":float(1-np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)) if len(y)>1 and np.var(y)>0 else None,
      "log_MAE_USD":float(np.abs(np.log1p(y)-np.log1p(p)).mean()),
      "within20":float(np.mean(relative<=.2)),
      "within30":float(np.mean(relative<=.3)),
      "within50":float(np.mean(relative<=.5)),
      "unique_predictions":int(len(np.unique(np.round(p))))}

def main():
    d=load()
    frozen=pd.read_csv(ROOT / "audit_inputs/50_200m_ml_specialist/accepted/candidate_predictions.csv")
    assert len(frozen)==912
    assert int((frozen.actual_band==frozen.candidate_pred_new).sum())==830
    assigned=frozen.loc[frozen.candidate_pred_new=="50-200M",["disasterNumber"]].copy()
    assert len(assigned)==13
    d=d.merge(assigned.assign(routed=True),on="disasterNumber",how="left",validate="one_to_one")
    rows=[]
    for year in sorted(d.loc[d.routed==True,"fyDeclared"].unique()):
        train=d[d.fyDeclared!=year]
        test=d[(d.fyDeclared==year)&(d.routed==True)].copy()
        selected,_=select(train)
        m=float(train.loc[(train.target_m>=LO)&(train.target_m<HI),"target_m"].median())
        b=float(train.loc[(train.target_m>=20)&(train.target_m<300),"target_m"].median())
        # Mask test targets. Predictions must be computable for a routed case without knowing its true band.
        masked=test.copy()
        masked["target_m"]=np.nan
        raw=candidate_preds(train,masked)[selected["candidate"]]
        final=np.clip(m+selected["alpha"]*(raw-m),LO,HI-1e-6)
        for (_,record),prediction in zip(test.iterrows(),final):
            rows.append({"disasterNumber":int(record.disasterNumber),"fyDeclared":int(year),
                "state":record.state,"incidentType":record.incidentType,
                "actual_m":float(record.target_m),"median_m":m,"broad_median_m":b,
                "hybrid_m":float(prediction),"candidate":selected["candidate"],
                "alpha":float(selected["alpha"]),"correct_route":bool(LO<=record.target_m<HI)})
        print(f"Completed routed FY {year}",flush=True)
    p=pd.DataFrame(rows).sort_values(["fyDeclared","disasterNumber"])
    assert len(p)==13 and int(p.correct_route.sum())==12
    actual=p.actual_m.to_numpy()
    report={"routed_n":len(p),"correctly_routed_n":int(p.correct_route.sum()),
       "median":metrics(actual,p.median_m),"broad_median":metrics(actual,p.broad_median_m),
       "hybrid":metrics(actual,p.hybrid_m),
       "frozen_router_correct":830,"frozen_router_total":912,
       "method":"Outer LFYO; selected method and shrinkage by inner LFYO within outer-training; test target explicitly hidden for prediction."}
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/"routed_predictions.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("BEGIN_ROUTED_METRICS_JSON",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print("END_ROUTED_METRICS_JSON",flush=True)
    print("BEGIN_ROUTED_CASES",flush=True)
    print(p.to_string(index=False),flush=True)
    print("END_ROUTED_CASES",flush=True)

if __name__=="__main__":
    main()
