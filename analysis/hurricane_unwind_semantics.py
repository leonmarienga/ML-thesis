#!/usr/bin/env python3
"""
Target-blind Hurricane Mission Assignment closeout/unwind semantics audit.

Question
--------
Can non-financial Mission Assignment amendment/closeout structure identify
"large operational response, heavily unwound later" cases such as Sandy NJ,
without using obligation amounts as predictors?

Protocol
--------
1. Build text/structural features for ALL Hurricane declarations first.
2. Redact digits/currency-scale terms from text before semantic matching.
3. Use only non-financial fields for candidate predictors:
   maAmendNumber, agencyId, supportFunction, maType, priority,
   assistanceRequested, statementOfWork.
4. Only AFTER feature construction, attach funding and a separate financial
   target-reconstruction audit. Financial values are diagnostic only.
5. This is exploratory/retrospective; no router change is made in this script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mission_semantic_audit import (
    fetch_all_mission_assignments,
    normalize_master,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "hurricane_unwind_semantics"
OUT.mkdir(parents=True, exist_ok=True)

PATTERNS = {
    "deobligation": r"\b(de[- ]?obligat\w*|deobligat\w*)\b",
    "closeout": r"\b(close[- ]?out|closeout|closing|final billing|final bill)\b",
    "unused_excess": r"\b(unused|unexpended|excess|remaining)\s+(funds?|balance|amount)\b|\b(return|release)\s+(unused|excess|remaining)?\s*(funds?|balance)\b",
    "reduce_reconcile": r"\b(reduc\w*|reconcil\w*|adjust\w*)\s+(funding|funds?|obligation|balance|estimate)\b",
    "cancel_terminate": r"\b(cancel\w*|terminat\w*|withdraw\w*)\b",
}

def redact_text(x: object) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", s)
    s = re.sub(r"\$|usd|dollars?|million|billion|thousand", " ", s)
    s = re.sub(r"\d+(?:[.,]\d+)*", " ", s)
    s = re.sub(r"[^a-z\s/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(np.nan, index=s.index)
    return 0.67448975 * (x - med) / mad

def main():
    master = normalize_master(pd.read_excel(MASTER))
    hurricanes = master[master["incidentType"] == "Hurricane"].copy()
    hids = set(hurricanes["disasterNumber"].dropna().astype(int))

    ma = fetch_all_mission_assignments().copy()
    ma["disasterNumber"] = pd.to_numeric(ma["disasterNumber"], errors="coerce").astype("Int64")
    x = ma[ma["disasterNumber"].isin(hids)].copy()

    x["amend_num"] = pd.to_numeric(x["maAmendNumber"], errors="coerce").fillna(0).clip(lower=0)
    x["is_amendment"] = (x["amend_num"] > 0).astype(int)
    x["action_num"] = pd.to_numeric(x["actionId"], errors="coerce").fillna(-1)

    x["clean_text"] = (
        x["assistanceRequested"].fillna("").astype(str)
        + " "
        + x["statementOfWork"].fillna("").astype(str)
    ).map(redact_text)

    for name, pat in PATTERNS.items():
        x[f"txt_{name}"] = x["clean_text"].str.contains(pat, regex=True, na=False).astype(int)
    text_flags = [f"txt_{k}" for k in PATTERNS]
    x["txt_any_unwind"] = x[text_flags].max(axis=1).astype(int)
    x["txt_any_unwind_amend"] = (x["txt_any_unwind"] & x["is_amendment"]).astype(int)

    # Per-mission structure from non-financial fields only.
    x = x.sort_values(["disasterNumber", "maId", "amend_num", "action_num"])
    gm = x.groupby(["disasterNumber", "maId"], dropna=False)
    per = gm.agg(
        mission_rows=("maId", "size"),
        mission_max_amend=("amend_num", "max"),
        mission_any_unwind=("txt_any_unwind", "max"),
        mission_unwind_rows=("txt_any_unwind", "sum"),
        mission_unwind_amend_rows=("txt_any_unwind_amend", "sum"),
    ).reset_index()
    per["mission_amended"] = (per["mission_max_amend"] >= 1).astype(int)
    per["mission_amend2plus"] = (per["mission_max_amend"] >= 2).astype(int)
    per["mission_amend5plus"] = (per["mission_max_amend"] >= 5).astype(int)

    # Latest row per mission, still non-financial.
    latest = gm.tail(1).copy()
    latest_cols = ["disasterNumber", "maId", "txt_any_unwind"] + text_flags
    latest = latest[latest_cols].rename(columns={
        "txt_any_unwind": "latest_any_unwind",
        **{c: f"latest_{c}" for c in text_flags},
    })
    per = per.merge(latest, on=["disasterNumber", "maId"], how="left")

    g = per.groupby("disasterNumber")
    feat = g.agg(
        unwind_unique_missions=("maId", "nunique"),
        unwind_amended_missions=("mission_amended", "sum"),
        unwind_amend2plus_missions=("mission_amend2plus", "sum"),
        unwind_amend5plus_missions=("mission_amend5plus", "sum"),
        unwind_missions_any_text=("mission_any_unwind", "sum"),
        unwind_total_text_rows=("mission_unwind_rows", "sum"),
        unwind_amend_text_rows=("mission_unwind_amend_rows", "sum"),
        unwind_latest_missions_any_text=("latest_any_unwind", "sum"),
        unwind_max_amendment=("mission_max_amend", "max"),
        unwind_mean_max_amendment=("mission_max_amend", "mean"),
    ).reset_index()

    denom = feat["unwind_unique_missions"].replace(0, np.nan)
    feat["unwind_amended_mission_share"] = feat["unwind_amended_missions"] / denom
    feat["unwind_amend2plus_mission_share"] = feat["unwind_amend2plus_missions"] / denom
    feat["unwind_amend5plus_mission_share"] = feat["unwind_amend5plus_missions"] / denom
    feat["unwind_mission_text_share"] = feat["unwind_missions_any_text"] / denom
    feat["unwind_latest_text_share"] = feat["unwind_latest_missions_any_text"] / denom

    # Pattern-specific mission shares (any row and latest row).
    for name in PATTERNS:
        c = f"txt_{name}"
        anym = x.groupby(["disasterNumber", "maId"])[c].max().groupby("disasterNumber").sum()
        latc = f"latest_{c}"
        latm = per.groupby("disasterNumber")[latc].sum()
        feat = feat.merge(anym.rename(f"unwind_{name}_missions"), on="disasterNumber", how="left")
        feat = feat.merge(latm.rename(f"unwind_latest_{name}_missions"), on="disasterNumber", how="left")
        feat[f"unwind_{name}_mission_share"] = feat[f"unwind_{name}_missions"] / denom
        feat[f"unwind_latest_{name}_share"] = feat[f"unwind_latest_{name}_missions"] / denom

    # Attach metadata only after target-blind features exist.
    out = hurricanes[[
        "disasterNumber", "state", "fyDeclared", "totalObligatedFunding",
        "missionAssignmentCount"
    ]].merge(feat, on="disasterNumber", how="left")
    out.to_csv(OUT / "all108_unwind_features.csv", index=False)

    # Robust target-blind ranks for candidate non-financial features.
    candidate = [
        "unwind_amended_mission_share",
        "unwind_amend2plus_mission_share",
        "unwind_amend5plus_mission_share",
        "unwind_mission_text_share",
        "unwind_latest_text_share",
        "unwind_max_amendment",
        "unwind_mean_max_amendment",
    ] + [
        f"unwind_{name}_mission_share" for name in PATTERNS
    ] + [
        f"unwind_latest_{name}_share" for name in PATTERNS
    ]
    for c in candidate:
        out[f"{c}_robust_z"] = robust_z(out[c])
        out[f"{c}_rank_desc"] = out[c].rank(method="min", ascending=False)
    out.to_csv(OUT / "all108_unwind_features_ranked.csv", index=False)

    # Financial reconstruction: audit only, never a predictor.
    x["obligation_num"] = pd.to_numeric(x["obligationAmount"], errors="coerce").fillna(0.0)
    fin = x.groupby("disasterNumber").agg(
        audit_positive_obligation=("obligation_num", lambda s: float(s[s > 0].sum())),
        audit_negative_obligation=("obligation_num", lambda s: float(s[s < 0].sum())),
        audit_net_obligation=("obligation_num", "sum"),
    ).reset_index()
    fin["audit_deobligation_fraction_of_positive"] = (
        (-fin["audit_negative_obligation"]) /
        fin["audit_positive_obligation"].replace(0, np.nan)
    )
    fin["audit_net_to_positive_ratio"] = (
        fin["audit_net_obligation"] /
        fin["audit_positive_obligation"].replace(0, np.nan)
    )
    audit = out.merge(fin, on="disasterNumber", how="left")
    audit["audit_target_delta"] = audit["audit_net_obligation"] - audit["totalObligatedFunding"]
    audit.to_csv(OUT / "all108_unwind_with_financial_audit.csv", index=False)

    focus_ids = [4085, 4086, 4611, 4671, 4337, 4393, 4559, 4673, 4830]
    focus = audit[audit["disasterNumber"].isin(focus_ids)].copy()
    focus.to_csv(OUT / "focus_cases.csv", index=False)

    # Sanitized matching text examples for Sandy and Ida only; no amounts.
    samples = x[
        x["disasterNumber"].isin([4085, 4086, 4611])
        & (x["txt_any_unwind"] == 1)
    ][[
        "disasterNumber", "maId", "maAmendNumber", "agencyId",
        "supportFunction", "maType", "priority", "clean_text"
    ] + text_flags].copy()
    samples.to_csv(OUT / "focus_unwind_text_samples.csv", index=False)

    summary = {
        "hurricane_declarations": int(len(out)),
        "candidate_nonfinancial_features": candidate,
        "focus": {},
        "important_method_note": (
            "All candidate unwind features are built before funding is attached. "
            "Obligation amounts are used only in the separate financial audit."
        ),
    }

    for dn in [4085, 4086, 4611]:
        r = audit[audit["disasterNumber"] == dn].iloc[0]
        summary["focus"][str(dn)] = {
            "state": r["state"],
            "funding_for_audit_only": float(r["totalObligatedFunding"]),
            "positive_obligation_audit": float(r["audit_positive_obligation"]),
            "negative_obligation_audit": float(r["audit_negative_obligation"]),
            "deobligation_fraction_audit": float(r["audit_deobligation_fraction_of_positive"]),
            "unwind_mission_text_share": float(r["unwind_mission_text_share"]),
            "unwind_latest_text_share": float(r["unwind_latest_text_share"]),
            "amended_mission_share": float(r["unwind_amended_mission_share"]),
            "amend5plus_mission_share": float(r["unwind_amend5plus_mission_share"]),
            "max_amendment": float(r["unwind_max_amendment"]),
            "feature_ranks": {
                c: (
                    int(r[f"{c}_rank_desc"])
                    if pd.notna(r[f"{c}_rank_desc"]) else None
                )
                for c in candidate
            },
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Hurricane Mission Assignment unwind-semantics audit",
        "",
        f"- Hurricane declarations enriched target-blind: **{len(out)}**",
        "- Router is **not changed** by this audit.",
        "- Financial amounts are audit-only.",
        "",
        "## Focus cases",
        "",
        "| FEMA | State | Positive obligations | Deobligations | Deobligation fraction | Unwind-text mission share | Latest unwind share | Amended mission share |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dn in [4086, 4085, 4611]:
        r = summary["focus"][str(dn)]
        md.append(
            f"| {dn} | {r['state']} | ${r['positive_obligation_audit']:,.0f} | "
            f"${-r['negative_obligation_audit']:,.0f} | "
            f"{r['deobligation_fraction_audit']:.1%} | "
            f"{r['unwind_mission_text_share']:.1%} | "
            f"{r['unwind_latest_text_share']:.1%} | "
            f"{r['amended_mission_share']:.1%} |"
        )

    md += [
        "",
        "## Sandy NJ target-blind feature ranks among all Hurricanes",
        "",
    ]
    sj = summary["focus"]["4086"]
    for c, rank in sj["feature_ranks"].items():
        md.append(f"- {c}: rank **{rank}/{len(out)}**")

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
