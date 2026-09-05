from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import multi_direction_high_router_ensemble as md

OUT = Path("nested_rl_referee_high_router_results")
OUT.mkdir(exist_ok=True)

BANDS=[3,4,5]

md.BIN_CFG["iterations"]=100
md.MULTI_CFG["iterations"]=130

ACTIONS=[
    "trust_bottom",
    "trust_top",
    "trust_sifter",
    "majority",
    "max_band",
    "min_band",
]

ALPHA=0.25
EPOCHS=35
GAMMA=0.0

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
        "pass80":bool(np.all(rec>=.8)),
    }

def majority(a,b,c):
    vals=[int(a),int(b),int(c)]
    counts={v:vals.count(v) for v in set(vals)}
    m=max(counts.values())
    winners=[v for v,k in counts.items() if k==m]
    return winners[0] if len(winners)==1 else int(c)

def action_prediction(action,a,b,c):
    if action=="trust_bottom": return int(a)
    if action=="trust_top": return int(b)
    if action=="trust_sifter": return int(c)
    if action=="majority": return majority(a,b,c)
    if action=="max_band": return max(int(a),int(b),int(c))
    if action=="min_band": return min(int(a),int(b),int(c))
    raise ValueError(action)

def reward(actual,pred):
    actual=int(actual); pred=int(pred)
    if pred==actual:
        return 2.0 if actual==5 else 1.0
    dist=abs(pred-actual)
    r=-1.0*dist
    if actual==5 and pred<5:
        r-=2.0
    if actual<5 and pred==5:
        r-=1.25
    return float(r)

def expert_predictions(train,test,nums,cats):
    a,_=md.bottom_up_predict(train,test,nums,cats)
    b,_=md.top_down_predict(train,test,nums,cats)
    c,_=md.sifter_predict(train,test,nums,cats)
    return np.asarray(a,int),np.asarray(b,int),np.asarray(c,int)

def build_inner_oof(outer_train,nums,cats):
    rows=[]
    for meta_year in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr=outer_train[outer_train["fyDeclared"].astype(int)!=meta_year].copy()
        va=outer_train[outer_train["fyDeclared"].astype(int)==meta_year].copy()
        if va.empty: continue
        a,b,c=expert_predictions(tr,va,nums,cats)
        for j,(_,r) in enumerate(va.iterrows()):
            rows.append({
                "row_index":int(r.name),
                "fyDeclared":int(r.fyDeclared),
                "incidentType":str(r.incidentType).upper().strip(),
                "actual":int(r.band),
                "bottom":int(a[j]),
                "top":int(b[j]),
                "sifter":int(c[j]),
            })
    return pd.DataFrame(rows)

def state_keys(row):
    a,b,c=int(row["bottom"]),int(row["top"]),int(row["sifter"])
    incident=str(row["incidentType"]).upper().strip()
    agree=int(a==b==c)
    spread=max(a,b,c)-min(a,b,c)
    votes5=int(a==5)+int(b==5)+int(c==5)
    return [
        ("full",a,b,c,incident),
        ("triplet",a,b,c),
        ("pattern",agree,spread,votes5),
        ("global",),
    ]

def train_bandit(inner):
    Q=defaultdict(lambda: {a:0.0 for a in ACTIONS})
    N=defaultdict(lambda: {a:0 for a in ACTIONS})
    records=inner.to_dict("records")
    for _ in range(EPOCHS):
        for r in records:
            keys=state_keys(r)
            for action in ACTIONS:
                pred=action_prediction(action,r["bottom"],r["top"],r["sifter"])
                rew=reward(r["actual"],pred)
                for key in keys:
                    old=Q[key][action]
                    Q[key][action]=old+ALPHA*(rew-old)
                    N[key][action]+=1
    return Q,N

def choose_action(row,Q,N):
    for key in state_keys(row):
        counts=N.get(key,{})
        if counts and max(counts.values())>0:
            qs=Q[key]
            pref={a:i for i,a in enumerate(["majority","trust_bottom","trust_sifter","trust_top","max_band","min_band"])}
            best=sorted(ACTIONS,key=lambda a:(-qs[a],pref[a]))[0]
            return best,key,{a:float(qs[a]) for a in ACTIONS}
    return "majority",("fallback",),{a:0.0 for a in ACTIONS}

def main():
    d=base.load_master()
    h=d[d["band"].isin(BANDS)].copy().reset_index(drop=True)
    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in md.SAFE_CATS if c in h.columns]

    bu=np.full(len(h),-99,int)
    td=np.full(len(h),-99,int)
    sf=np.full(len(h),-99,int)
    maj=np.full(len(h),-99,int)
    rl=np.full(len(h),-99,int)

    rows=[]
    fold_rows=[]

    for outer_year in sorted(h["fyDeclared"].astype(int).unique()):
        tr_mask=h["fyDeclared"].astype(int)!=outer_year
        te_mask=~tr_mask
        outer_train=h.loc[tr_mask].copy()
        outer_test=h.loc[te_mask].copy()

        inner=build_inner_oof(outer_train,nums,cats)
        Q,N=train_bandit(inner)

        a,b,c=expert_predictions(outer_train,outer_test,nums,cats)
        idx=np.flatnonzero(te_mask.to_numpy())

        fold_actions=[]
        for j,ii in enumerate(idx):
            r=h.iloc[ii]
            mj=majority(a[j],b[j],c[j])
            srow={
                "bottom":int(a[j]),"top":int(b[j]),"sifter":int(c[j]),
                "incidentType":str(r.incidentType).upper().strip(),
            }
            action,key,qvals=choose_action(srow,Q,N)
            pred=action_prediction(action,a[j],b[j],c[j])

            bu[ii]=int(a[j]); td[ii]=int(b[j]); sf[ii]=int(c[j])
            maj[ii]=int(mj); rl[ii]=int(pred)

            rows.append({
                "outer_year":int(outer_year),
                "disasterNumber":int(r.disasterNumber),
                "state":str(r.state),
                "incidentType":str(r.incidentType),
                "target":float(r.target),
                "band":int(r.band),
                "bottom_up":int(a[j]),
                "top_down":int(b[j]),
                "sifter":int(c[j]),
                "majority_vote":int(mj),
                "rl_action":action,
                "rl_state_level":str(key[0]),
                "rl_prediction":int(pred),
                "q_values":json.dumps(qvals,sort_keys=True),
            })
            fold_actions.append(action)

        fold_rows.append({
            "outer_year":int(outer_year),
            "inner_rows":int(len(inner)),
            "inner_band_counts":json.dumps(inner["actual"].value_counts().sort_index().to_dict()),
            "action_counts":json.dumps(pd.Series(fold_actions).value_counts().to_dict()),
        })

    y=h["band"].astype(int).to_numpy()
    metrics={
        "bottom_up":metric(y,bu),
        "top_down":metric(y,td),
        "sifter":metric(y,sf),
        "majority_vote":metric(y,maj),
        "nested_rl_referee":metric(y,rl),
    }

    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_rl_oof_predictions.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/"outer_fold_rl_training_summary.csv",index=False)

    actual5=out[out.band==5].copy()
    false5=out[(out.band!=5)&(out.rl_prediction==5)].copy()
    rescued=out[(out.rl_prediction==out.band)&(out.majority_vote!=out.band)].copy()
    harmed=out[(out.rl_prediction!=out.band)&(out.majority_vote==out.band)].copy()

    summary={
        "purpose":"Fully nested contextual-bandit/Q-learning referee over the three high-value routing experts on all 55 cases including Biological.",
        "algorithm":{
            "type":"offline full-information contextual bandit / one-step Q-learning",
            "actions":ACTIONS,
            "alpha":ALPHA,
            "epochs":EPOCHS,
            "gamma":GAMMA,
            "state_backoff":["router triplet + incident type","router triplet","agreement/spread/extreme-vote pattern","global"],
            "reward":{
                "correct_non_extreme":"+1",
                "correct_extreme":"+2",
                "wrong_per_band_distance":"-1 x distance",
                "extra_true_extreme_miss":"-2",
                "extra_false_extreme_alarm":"-1.25",
            }
        },
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "metrics":metrics,
        "actual_500m_plus":actual5[
            ["disasterNumber","state","incidentType","target","bottom_up","top_down","sifter",
             "majority_vote","rl_action","rl_state_level","rl_prediction","q_values"]
        ].to_dict("records"),
        "rl_false_500m_plus":false5[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","rl_action","rl_prediction"]
        ].to_dict("records"),
        "rl_rescued_vs_majority":rescued[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","rl_action","rl_prediction"]
        ].to_dict("records"),
        "rl_harmed_vs_majority":harmed[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","rl_action","rl_prediction"]
        ].to_dict("records"),
        "outer_folds":fold_rows,
        "guardrails":[
            "All 55 high-value cases including Biological are evaluated.",
            "The outer fiscal year is excluded from every expert and every RL-referee training operation.",
            "RL training states are generated only from an additional leave-fiscal-year-out loop inside each outer training set.",
            "The reward schedule and action set are fixed before outer evaluation; they are not tuned against outer OOF results.",
            "Funding-derived eventScale is excluded from the routing experts.",
            "The RL referee receives only expert predictions, their disagreement pattern, and incident type; no funding value or funding-derived feature is used.",
            "For outer FY2020, the RL referee has zero high-value Biological examples in training, so the Biological regime remains a genuine temporal cold-start test.",
            "Because this is a one-step referee decision, gamma is zero; this is appropriately a contextual-bandit form of reinforcement learning rather than a deep sequential DQN."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
