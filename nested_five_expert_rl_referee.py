from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import multi_direction_high_router_ensemble as md
import nested_ovr_pairwise_high_router as op

OUT=Path("nested_five_expert_rl_referee_results")
OUT.mkdir(exist_ok=True)

BANDS=[3,4,5]

# Lighter fixed expert configs for deep nesting; no outer-result tuning.
md.BIN_CFG["iterations"]=85
md.MULTI_CFG["iterations"]=110
op.CFG["iterations"]=100

ACTIONS=[
    "trust_bottom","trust_top","trust_sifter","trust_ovr","trust_pairwise",
    "majority5","max_band","min_band"
]
ALPHA=.25
EPOCHS=30

def metric(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    rec=recall_score(y,p,labels=BANDS,average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=BANDS)
    return {
        "correct3":int(cm[0,0]),"n3":int((y==3).sum()),"r3":float(rec[0]),
        "correct4":int(cm[1,1]),"n4":int((y==4).sum()),"r4":float(rec[1]),
        "correct5":int(cm[2,2]),"n5":int((y==5).sum()),"r5":float(rec[2]),
        "min_recall":float(rec.min()),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "confusion_matrix":cm.tolist(),
        "pass80":bool(np.all(rec>=.8))
    }

def majority5(vals):
    vals=[int(v) for v in vals]
    counts={v:vals.count(v) for v in set(vals)}
    m=max(counts.values())
    winners=[v for v,k in counts.items() if k==m]
    if len(winners)==1:return winners[0]
    # fixed tie break: pairwise, then OVR, then sifter
    for j in [4,3,2,0,1]:
        if vals[j] in winners:return vals[j]
    return winners[0]

def action_pred(action,vals):
    b,t,s,o,p=[int(v) for v in vals]
    if action=="trust_bottom":return b
    if action=="trust_top":return t
    if action=="trust_sifter":return s
    if action=="trust_ovr":return o
    if action=="trust_pairwise":return p
    if action=="majority5":return majority5(vals)
    if action=="max_band":return max(vals)
    if action=="min_band":return min(vals)
    raise ValueError(action)

def reward(actual,pred):
    actual=int(actual);pred=int(pred)
    if actual==pred:return 2.25 if actual==5 else (1.25 if actual==4 else 1.0)
    r=-float(abs(actual-pred))
    if actual==5 and pred<5:r-=2.5
    if actual<5 and pred==5:r-=1.5
    return r

def experts(train,test,nums,cats):
    b,_=md.bottom_up_predict(train,test,nums,cats)
    t,_=md.top_down_predict(train,test,nums,cats)
    s,_=md.sifter_predict(train,test,nums,cats)
    o,_,_,_=op.ovr_predict(train,test,nums,cats)
    p,_,_,_=op.pair_predict(train,test,nums,cats)
    return tuple(np.asarray(x,int) for x in [b,t,s,o,p])

def inner_rows(outer_train,nums,cats):
    rows=[]
    for yr in sorted(outer_train.fyDeclared.astype(int).unique()):
        tr=outer_train[outer_train.fyDeclared.astype(int)!=yr].copy()
        va=outer_train[outer_train.fyDeclared.astype(int)==yr].copy()
        if va.empty:continue
        preds=experts(tr,va,nums,cats)
        for j,(_,r) in enumerate(va.iterrows()):
            vals=[int(x[j]) for x in preds]
            rows.append({
                "actual":int(r.band),
                "incidentType":str(r.incidentType).upper().strip(),
                "vals":vals
            })
    return rows

def state_keys(vals,incident):
    vals=tuple(int(v) for v in vals)
    counts=tuple(vals.count(b) for b in BANDS)
    spread=max(vals)-min(vals)
    topvotes=vals.count(5)
    agreement=max(counts)
    return [
        ("full",)+vals+(incident,),
        ("votes",)+vals,
        ("pattern",)+counts+(spread,topvotes,agreement),
        ("global",)
    ]

def train_q(rows):
    Q=defaultdict(lambda:{a:0.0 for a in ACTIONS})
    N=defaultdict(lambda:{a:0 for a in ACTIONS})
    for _ in range(EPOCHS):
        for r in rows:
            for action in ACTIONS:
                rew=reward(r["actual"],action_pred(action,r["vals"]))
                for key in state_keys(r["vals"],r["incidentType"]):
                    Q[key][action]+=ALPHA*(rew-Q[key][action])
                    N[key][action]+=1
    return Q,N

def choose(vals,incident,Q,N):
    pref={a:i for i,a in enumerate([
        "majority5","trust_pairwise","trust_ovr","trust_bottom",
        "trust_sifter","trust_top","max_band","min_band"
    ])}
    for key in state_keys(vals,incident):
        if key in N and max(N[key].values())>0:
            q=Q[key]
            a=sorted(ACTIONS,key=lambda x:(-q[x],pref[x]))[0]
            return a,key,{k:float(v) for k,v in q.items()}
    return "majority5",("fallback",),{a:0.0 for a in ACTIONS}

def main():
    d=base.load_master()
    h=d[d.band.isin(BANDS)].copy().reset_index(drop=True)
    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in md.SAFE_CATS if c in h.columns]

    pred_arrays={k:np.full(len(h),-99,int) for k in
                 ["bottom","top","sifter","ovr","pairwise","majority5","rl"]}
    rows=[];folds=[]

    for outer in sorted(h.fyDeclared.astype(int).unique()):
        trm=h.fyDeclared.astype(int)!=outer
        tem=~trm
        train=h.loc[trm].copy();test=h.loc[tem].copy()

        ir=inner_rows(train,nums,cats)
        Q,N=train_q(ir)
        ep=experts(train,test,nums,cats)
        idx=np.flatnonzero(tem.to_numpy())
        action_counts=[]

        for j,ii in enumerate(idx):
            r=h.iloc[ii]
            vals=[int(x[j]) for x in ep]
            maj=majority5(vals)
            action,key,q=choose(vals,str(r.incidentType).upper().strip(),Q,N)
            rp=action_pred(action,vals)

            for name,val in zip(["bottom","top","sifter","ovr","pairwise"],vals):
                pred_arrays[name][ii]=val
            pred_arrays["majority5"][ii]=maj
            pred_arrays["rl"][ii]=rp
            action_counts.append(action)

            rows.append({
                "outer_year":int(outer),"disasterNumber":int(r.disasterNumber),
                "state":str(r.state),"incidentType":str(r.incidentType),
                "target":float(r.target),"band":int(r.band),
                "bottom":vals[0],"top":vals[1],"sifter":vals[2],
                "ovr":vals[3],"pairwise":vals[4],"majority5":maj,
                "rl_action":action,"rl_state_level":str(key[0]),
                "rl_prediction":rp,"q_values":json.dumps(q,sort_keys=True)
            })

        folds.append({
            "outer_year":int(outer),"inner_rows":len(ir),
            "action_counts":pd.Series(action_counts).value_counts().to_dict()
        })

    y=h.band.astype(int).to_numpy()
    metrics={k:metric(y,v) for k,v in pred_arrays.items()}
    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_five_expert_rl_oof.csv",index=False)

    actual5=out[out.band==5]
    summary={
        "purpose":"Fully nested RL referee combining five structurally different high-value routers: bottom-up, top-down, sifter, OVR, and pairwise.",
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "metrics":metrics,
        "actual_500m_plus":actual5[
            ["disasterNumber","state","incidentType","target","bottom","top",
             "sifter","ovr","pairwise","majority5","rl_action","rl_prediction","q_values"]
        ].to_dict("records"),
        "cases_any_expert_correct":int(((out[["bottom","top","sifter","ovr","pairwise"]]
            .to_numpy()==out.band.to_numpy()[:,None]).any(axis=1)).sum()),
        "cases_no_expert_correct":out[
            ~(out[["bottom","top","sifter","ovr","pairwise"]].to_numpy()
              ==out.band.to_numpy()[:,None]).any(axis=1)
        ][["disasterNumber","state","incidentType","target","band","bottom","top","sifter","ovr","pairwise"]].to_dict("records"),
        "outer_folds":folds,
        "guardrails":[
            "All 55 high-value cases including Biological are evaluated.",
            "Every outer fiscal year is excluded from every expert and RL training operation.",
            "Each RL training state is generated by experts fitted without that inner row's fiscal year.",
            "OVR and pairwise thresholds are selected only inside their current training support by leave-year validation.",
            "Reward/action/state definitions are fixed before outer evaluation.",
            "No funding-derived eventScale or held-out funding value enters any expert or referee.",
            "FY2020 Biological remains a genuine temporal cold-start."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
