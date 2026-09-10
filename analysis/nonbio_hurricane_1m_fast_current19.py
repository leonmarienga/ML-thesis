#!/usr/bin/env python3
"""
Fast strict-LFYO Hurricane $1M rescue cross-check using CURRENT_19 only.

Purpose: test whether Hurricane contains genuine lower-boundary signal without waiting
for MissionAssignment semantic enrichment. The accepted 814/912 predictions are frozen.
A Hurricane specialist may only promote accepted 100K-1M predictions to 1M-50M.
No >=50M prediction can change. Biological declarations are excluded.

For each outer fiscal year:
- train only on Hurricane rows in [100K, 50M) from all other years;
- generate inner LFYO probabilities across the outer-training years;
- select a probability threshold maximizing Hurricane two-band macro recall, subject
  to >=75% inner recall for genuine 100K-1M Hurricanes;
- fit on the complete outer-training set and apply only to accepted 814 Hurricane
  candidates currently predicted 100K-1M.

Variants: current19 logistic and current19 random forest.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mission_semantic_audit import CURRENT_19, normalize_master, normalize_model_frame, prep_pipeline
from nonbio_all_ranges import funding_band, valid_cols

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "hurricane_fast" / "accepted" / "integrated_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_hurricane_1m_fast_current19"
OUT.mkdir(parents=True, exist_ok=True)
HAZARD = "Hurricane"


def fit_model(train, features, kind, seed):
    t = train[(train.incidentType == HAZARD) & (train.target_clean >= 100_000) & (train.target_clean < 50_000_000)].copy()
    y = (t.target_clean >= 1_000_000).astype(int)
    X = normalize_model_frame(t[features])
    if kind == "rf":
        model = RandomForestClassifier(n_estimators=700, random_state=seed, class_weight="balanced_subsample", max_features="sqrt", min_samples_leaf=1, n_jobs=1)
    else:
        model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def inner_scores(train, features, kind, outer_fy):
    chunks = []
    h = train[(train.incidentType == HAZARD) & (train.target_clean >= 100_000) & (train.target_clean < 50_000_000)].copy()
    for fy in sorted(h.fyDeclared.astype(int).unique()):
        tr = train[train.fyDeclared.astype(int) != fy].copy()
        te = h[h.fyDeclared.astype(int) == fy].copy()
        st = tr[(tr.incidentType == HAZARD) & (tr.target_clean >= 100_000) & (tr.target_clean < 50_000_000)]
        ytr = (st.target_clean >= 1_000_000).astype(int)
        if te.empty or len(st) < 10 or ytr.nunique() < 2:
            continue
        m = fit_model(tr, features, kind, 700000 + int(outer_fy) * 100 + int(fy) + (10000 if kind == "rf" else 0))
        p = m.predict_proba(normalize_model_frame(te[features]))[:, 1]
        z = pd.DataFrame({"prob": p, "y": (te.target_clean.to_numpy() >= 1_000_000).astype(int)})
        chunks.append(z)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["prob", "y"])


def choose_threshold(scores, guard=0.75):
    if scores.empty:
        return 1.0, {"macro": np.nan, "low_recall": np.nan, "mid_recall": np.nan, "accuracy": np.nan}
    probs = scores.prob.to_numpy(float)
    grid = np.unique(np.r_[0.05, np.arange(0.10, 0.96, 0.02), 0.99, probs])
    best = None
    y = scores.y.to_numpy(int)
    for th in grid:
        pred = (probs >= th).astype(int)
        low = y == 0; mid = y == 1
        lr = float((pred[low] == 0).mean()) if low.any() else np.nan
        mr = float((pred[mid] == 1).mean()) if mid.any() else np.nan
        if np.isfinite(lr) and lr < guard:
            continue
        macro = float(np.nanmean([lr, mr]))
        acc = float((pred == y).mean())
        key = (macro, acc, float(th))
        if best is None or key > best[0]:
            best = (key, float(th), {"macro": macro, "low_recall": lr, "mid_recall": mr, "accuracy": acc})
    if best is None:
        return 1.0, {"macro": np.nan, "low_recall": np.nan, "mid_recall": np.nan, "accuracy": np.nan}
    return best[1], best[2]


def full_metrics(df, col):
    bands = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
    per = {}; rec = []
    correct = int((df[col] == df.actual_band).sum())
    for b in bands:
        m = df.actual_band == b; n = int(m.sum()); c = int((df.loc[m, col] == b).sum())
        r = c / n if n else np.nan
        per[b] = {"correct": c, "total": n, "recall": r}
        if n: rec.append(r)
    return {"correct": correct, "total": len(df), "accuracy": correct/len(df), "macro": float(np.mean(rec)), "per_band": per}


def main():
    d = normalize_master(pd.read_excel(MASTER))
    d["target_clean"] = pd.to_numeric(d.totalObligatedFunding, errors="coerce").fillna(0).clip(lower=0)
    d["actual_band"] = d.target_clean.map(funding_band)
    d = d[d.incidentType != "Biological"].copy().reset_index(drop=True)
    d["disasterNumber"] = d.disasterNumber.astype(int)
    features = valid_cols(d, CURRENT_19)

    a = pd.read_csv(ACCEPTED)
    a["disasterNumber"] = a.disasterNumber.astype(int)
    base_col = "storm_integrated_pred"
    assert base_col in a.columns
    keep = ["disasterNumber", base_col]
    x = d.merge(a[keep], on="disasterNumber", how="left", validate="one_to_one")
    assert x[base_col].notna().all()
    assert len(x) == 912

    outputs = {}
    thresholds = []
    for kind in ["log", "rf"]:
        pred = x[base_col].astype(str).copy()
        probs_all = np.full(len(x), np.nan)
        for outer_fy in sorted(x.fyDeclared.astype(int).unique()):
            train = x[x.fyDeclared.astype(int) != outer_fy].copy()
            test_idx = np.flatnonzero(x.fyDeclared.astype(int).to_numpy() == outer_fy)
            scores = inner_scores(train, features, kind, int(outer_fy))
            th, diag = choose_threshold(scores, guard=0.75)
            thresholds.append({"variant": kind, "outer_fy": int(outer_fy), "threshold": float(th), **diag})

            st = train[(train.incidentType == HAZARD) & (train.target_clean >= 100_000) & (train.target_clean < 50_000_000)]
            ytr = (st.target_clean >= 1_000_000).astype(int)
            if len(st) < 10 or ytr.nunique() < 2:
                continue
            model = fit_model(train, features, kind, 800000 + int(outer_fy) + (10000 if kind == "rf" else 0))
            cand = [i for i in test_idx if x.iloc[i].incidentType == HAZARD and str(x.iloc[i][base_col]) == "100K-1M"]
            if not cand:
                continue
            p = model.predict_proba(normalize_model_frame(x.iloc[cand][features]))[:, 1]
            probs_all[cand] = p
            promote = np.asarray(cand)[p >= th]
            pred.iloc[promote] = "1M-50M"

        col = f"hurricane_{kind}_pred"
        out = x.copy(); out[col] = pred; out["specialist_prob"] = probs_all
        high = out.target_clean >= 50_000_000
        assert (out.loc[high, col] == out.loc[high, base_col]).all(), ">=50M actual prediction changed"
        changed = out[out[col] != out[base_col]].copy()
        assert ((changed.incidentType == HAZARD) & (changed[base_col] == "100K-1M") & (changed[col] == "1M-50M")).all()
        m = full_metrics(out, col)
        hm = {}
        for b in ["100K-1M", "1M-50M"]:
            q = out[(out.incidentType == HAZARD) & (out.actual_band == b)]
            hm[b] = {"correct": int((q[col] == b).sum()), "total": len(q)}
        outputs[kind] = {"metrics": m, "hurricane_boundary": hm, "promotions": len(changed), "changed": changed[["disasterNumber","state","fyDeclared","actual_band",base_col,col,"specialist_prob"]].to_dict(orient="records")}
        out.to_csv(OUT / f"{kind}_predictions.csv", index=False)

    base = full_metrics(x, base_col)
    pd.DataFrame(thresholds).to_csv(OUT / "thresholds.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"base": base, "variants": outputs}, indent=2), encoding="utf-8")

    md = ["# Fast CURRENT_19 Hurricane $1M rescue", "", "| Variant | Overall | Macro | 100K-1M | 1M-50M | Promotions | Hurricane low | Hurricane mid |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    bp = base["per_band"]
    md.append(f"| accepted_814 | {base['correct']}/912 ({base['accuracy']:.1%}) | {base['macro']:.1%} | {bp['100K-1M']['correct']}/108 ({bp['100K-1M']['recall']:.1%}) | {bp['1M-50M']['correct']}/82 ({bp['1M-50M']['recall']:.1%}) | 0 | - | - |")
    for kind, r in outputs.items():
        m=r['metrics']; p=m['per_band']; h=r['hurricane_boundary']
        md.append(f"| {kind} | {m['correct']}/912 ({m['accuracy']:.1%}) | {m['macro']:.1%} | {p['100K-1M']['correct']}/108 ({p['100K-1M']['recall']:.1%}) | {p['1M-50M']['correct']}/82 ({p['1M-50M']['recall']:.1%}) | {r['promotions']} | {h['100K-1M']['correct']}/{h['100K-1M']['total']} | {h['1M-50M']['correct']}/{h['1M-50M']['total']} |")
        md += ["", f"## {kind} promotions", "", "| Disaster | State | FY | Actual | Base | New | P(mid) |", "|---:|---|---:|---|---|---|---:|"]
        for z in r['changed']:
            md.append(f"| {z['disasterNumber']} | {z['state']} | {z['fyDeclared']} | {z['actual_band']} | {z[base_col]} | {z[f'hurricane_{kind}_pred']} | {z['specialist_prob']:.4f} |")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
