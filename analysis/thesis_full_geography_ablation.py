#!/usr/bin/env python3
"""Test the page-complete retrospective features on shared episode-purged forward folds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from thesis_forward_models import ARMS, YEARS, fit_predict, read_table, score, split
from thesis_precoverage_ablation import COVERAGE_START

KEY = "disasterNumber"
ORIGINAL = Path("research/results/hazards/cohort_features.csv")
CONTRASTS = ("geography","verified_link_only")
SEED = 2609117


def shared_episodes(original, recovered):
    parents={int(number):int(number) for number in original.index}

    def find(key):
        while parents[key]!=key:
            parents[key]=parents[parents[key]]
            key=parents[key]
        return key

    for frame in (original,recovered):
        for group in frame.groupby("event_group").groups.values():
            first=int(group[0])
            for key in group[1:]:
                a,b=find(first),find(int(key))
                parents[max(a,b)]=min(a,b)
    return pd.Series({key:find(key) for key in parents},name="shared_episode")


def build_predictions(old, full):
    protected_columns=["fyDeclared","state","incidentType","start","end","declaration","target","actual_band"]
    pd.testing.assert_frame_equal(old[protected_columns],full[protected_columns],check_exact=True)
    groups=shared_episodes(old,full)
    source={"prior_pages":old.assign(event_group=groups),"all_pages":full.assign(event_group=groups)}
    ledger, cases=[],[]
    for year in YEARS:
        prior,test=split(source["prior_pages"].reset_index(),year)
        prior=prior[prior.declaration.ge(COVERAGE_START)]
        train_ids,held_ids=prior.disasterNumber,test.disasterNumber
        archived=old.loc[train_ids]
        restored=full.loc[train_ids]
        assert not set(archived.event_group)&set(old.loc[held_ids].event_group)
        assert not set(restored.event_group)&set(full.loc[held_ids].event_group)
        ledger.append(pd.concat([prior.assign(role="train"),test.assign(role="held")],ignore_index=True)[
            [KEY,"fyDeclared","event_group","role"]].assign(hold_fy=year))
        variants=[("metadata_reference","prior_pages","metadata"),
                  *((f"{scope}_{arm}",scope,arm) for scope in source for arm in CONTRASTS)]
        context=test[[KEY,"fyDeclared","incidentType","event_group","target","actual_band"]]
        for tag,scope,arm in variants:
            frame=source[scope]
            probabilities,amount=fit_predict(frame.loc[train_ids],frame.loc[held_ids],ARMS[arm],"extra_trees",6)
            output=context.assign(variant=tag,amount=amount)
            for band in range(6):
                output[f"p{band}"]=probabilities[:,band]
            cases.append(output)
    results=pd.concat(cases,ignore_index=True)
    assert results.groupby("variant").disasterNumber.nunique().eq(376).all()
    assert results.groupby("variant").actual_band.apply(lambda a:a.ge(3).sum()).eq(13).all()
    return results,pd.concat(ledger,ignore_index=True)


def metrics(cases):
    records=[]
    for variant,group in cases.groupby("variant"):
        for scope,frame in [("all",group),*group.groupby("fyDeclared")]:
            pm=frame[[f"p{i}" for i in range(6)]].to_numpy()
            actual=frame.actual_band.to_numpy()
            y=actual>=3
            high=pm[:,3:].sum(axis=1)
            scorecard=score(frame,frame.amount.to_numpy(),pm)
            high_mae=(frame.loc[y,"target"]-frame.loc[y,"amount"]).abs().mean()
            records.append({"variant":variant,"scope":scope,**scorecard,"tail_n":y.sum(),"tail_mae_usd":high_mae,
              "tail_ap":average_precision_score(y,high) if y.any() else np.nan,
              "tail_brier":np.mean((high-y)**2),"tail_true_alerts":int(((high>=.1)&y).sum()),
              "tail_false_alerts":int(((high>=.1)&~y).sum())})
    return pd.DataFrame(records)


def paired(cases):
    rows=[]
    for arm in CONTRASTS:
        a=cases[cases.variant.eq(f"prior_pages_{arm}")].sort_values(KEY).reset_index(drop=True)
        b=cases[cases.variant.eq(f"all_pages_{arm}")].sort_values(KEY).reset_index(drop=True)
        assert a.disasterNumber.equals(b.disasterNumber)
        n=len(a)
        groups=list(a.groupby("event_group").indices.values())
        indexes=[np.arange(n),*[np.concatenate([groups[j] for j in seq]) for seq in
                 np.random.default_rng(SEED).integers(0,len(groups),size=(1200,len(groups)))]]
        y=a.actual_band.ge(3).to_numpy()
        target=a.target.to_numpy()
        band=a.actual_band.to_numpy()
        effects=[]
        for d in (a,b):
            error=abs(d.amount.to_numpy()-target)
            p=d[[f"p{i}" for i in range(6)]].to_numpy()
            predicted=p.argmax(axis=1)
            correct=predicted==band
            prob=p[:,3:].sum(axis=1)
            def evaluate(i):
                confusion=np.bincount(6*band[i]+predicted[i],minlength=36).reshape(6,6)
                support=confusion.sum(axis=1)
                diagonal=confusion.diagonal()
                supported=support>0
                balanced=(diagonal[supported]/support[supported]).mean()
                f1_scale=support+confusion.sum(axis=0)
                macro=np.divide(2*diagonal,f1_scale,out=np.zeros(6),where=f1_scale>0).mean()
                return [correct[i].mean(),error[i].mean(),error[i][y[i]].mean() if y[i].any() else np.nan,
                        average_precision_score(y[i],prob[i]) if y[i].any() else np.nan,
                        balanced,macro,np.mean((prob[i]-y[i])**2)]
            effects.append(np.array([evaluate(i) for i in indexes]))
        delta=effects[1]-effects[0]
        descriptions=("accuracy","mae_usd","tail_mae_usd","tail_ap","balanced_accuracy","macro_f1","tail_brier")
        for j,description in enumerate(descriptions):
            lo,hi=np.nanquantile(delta[1:,j],[.025,.975])
            rows.append({"model":arm,"measure":description,"change":delta[0,j],"p025":lo,"p975":hi,
                         "episode_groups":len(groups)})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("recovered",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    old=read_table(ORIGINAL).set_index(KEY).sort_index()
    recovered=read_table(args.recovered).set_index(KEY).sort_index()
    cases,membership=build_predictions(old,recovered)
    for name,table in [("predictions.csv.gz",cases),("membership.csv.gz",membership),
                       ("metrics.csv",metrics(cases)),("paired.csv",paired(cases))]:
        kwargs={"compression":{"method":"gzip","mtime":0}} if name.endswith(".gz") else {}
        table.to_csv(args.output/name,index=False,**kwargs)
    print("Completed matched shared-episode comparisons for all six outer years.")


if __name__=="__main__":
    main()
