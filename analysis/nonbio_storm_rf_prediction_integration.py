#!/usr/bin/env python3
"""
Deterministic prediction-level integration of accepted 813 + Severe Storm current19 RF rescue.

Uses completed strict GitHub artifacts only. No model retraining or threshold reselection.
Safety checks require every Severe Storm promotion to match the accepted 813 low prediction
and prohibit changes to any actual or routed >=$50M case.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "audit_inputs" / "storm_rf_integration"
OUT = ROOT / "audit_outputs" / "storm_rf_prediction_integration"
OUT.mkdir(parents=True, exist_ok=True)

ACCEPTED = IN / "accepted" / "integrated_predictions.csv"
STORM = IN / "storm" / "storm_current_rf_predictions.csv"

BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]


def metrics(df, pred_col):
    per = {}
    recalls = []
    total_correct = 0
    for band in BANDS:
        m = df["actual_band"] == band
        n = int(m.sum())
        c = int((df.loc[m, pred_col] == band).sum())
        r = c / n if n else None
        per[band] = {"correct": c, "total": n, "recall": r}
        if r is not None:
            recalls.append(r)
        total_correct += c
    return {
        "overall_correct": total_correct,
        "overall_total": int(len(df)),
        "overall_accuracy": total_correct / len(df),
        "macro_recall": sum(recalls) / len(recalls),
        "per_band": per,
    }


def main():
    a = pd.read_csv(ACCEPTED)
    s = pd.read_csv(STORM)
    a["disasterNumber"] = a["disasterNumber"].astype(int)
    s["disasterNumber"] = s["disasterNumber"].astype(int)

    baseline_col = "integrated_pred"
    assert baseline_col in a.columns, "Accepted 813 artifact missing integrated_pred."

    changed = s[s["final_pred"] != s["base_pred"]].copy()
    assert len(changed) > 0, "Severe Storm RF made no promotions."
    assert set(changed["incidentType"]) == {"Severe Storm"}, "Non-Severe-Storm promotion found."
    assert ((changed["base_pred"] == "100K-1M") & (changed["final_pred"] == "1M-50M")).all(), \
        "Specialist made a non-one-way promotion."

    merged = changed[[
        "disasterNumber", "incidentType", "actual_band", "base_pred", "final_pred"
    ]].merge(
        a[[
            "disasterNumber", "actual_band", baseline_col,
            "root_actual_high", "root_pred_high"
        ]],
        on="disasterNumber",
        how="left",
        suffixes=("_storm", "_accepted"),
        validate="one_to_one",
    )

    assert merged[baseline_col].notna().all(), "Promotion absent from accepted 813 predictions."
    mismatches = merged[merged[baseline_col] != merged["base_pred"]]
    assert mismatches.empty, (
        "Accepted 813 prediction disagrees with Severe Storm base for promoted rows: "
        + str(mismatches[["disasterNumber", "base_pred", baseline_col]].to_dict(orient="records"))
    )
    assert (merged["root_actual_high"] == 0).all(), "Promotion touches actual >=$50M row."
    assert (merged["root_pred_high"] == 0).all(), "Promotion touches routed high row."

    out = a.copy()
    out["storm_integrated_pred"] = out[baseline_col]
    rescue_map = dict(zip(changed["disasterNumber"], changed["final_pred"]))
    mask = out["disasterNumber"].isin(rescue_map)
    out.loc[mask, "storm_integrated_pred"] = out.loc[mask, "disasterNumber"].map(rescue_map)

    high = out["root_actual_high"] == 1
    assert (out.loc[high, "storm_integrated_pred"] == out.loc[high, baseline_col]).all(), \
        "High-value prediction changed."

    base = metrics(out, baseline_col)
    new = metrics(out, "storm_integrated_pred")

    changed_eval = merged.copy()
    changed_eval["base_correct"] = changed_eval["base_pred"] == changed_eval["actual_band_storm"]
    changed_eval["storm_correct"] = changed_eval["final_pred"] == changed_eval["actual_band_storm"]

    out.to_csv(OUT / "integrated_predictions.csv", index=False)
    changed_eval.to_csv(OUT / "storm_promotions.csv", index=False)

    summary = {
        "source_runs": {
            "accepted_813_prediction_integration": 34437863627,
            "severe_storm_rf": 34438034102,
        },
        "promotion_count": int(len(changed_eval)),
        "promotion_disaster_numbers": changed_eval["disasterNumber"].astype(int).tolist(),
        "base_metrics": base,
        "integrated_metrics": new,
        "net_correct_gain": int(new["overall_correct"] - base["overall_correct"]),
        "macro_gain": float(new["macro_recall"] - base["macro_recall"]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Accepted 813 + Severe Storm RF prediction integration",
        "",
        "- Uses only completed strict GitHub workflow predictions.",
        "- No model retraining or threshold reselection.",
        "- All Severe Storm promotions match the accepted 813 low prediction.",
        "- No actual or routed >=$50M row is modified.",
        "",
        "| Variant | Overall | Macro | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in [("accepted_813", base), ("plus_storm_current_rf", new)]:
        pb = m["per_band"]
        md.append(
            f"| {name} | {m['overall_correct']}/{m['overall_total']} ({m['overall_accuracy']:.1%}) | "
            f"{m['macro_recall']:.1%} | "
            f"{pb['0-100K']['correct']}/{pb['0-100K']['total']} ({pb['0-100K']['recall']:.1%}) | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} ({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} ({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} ({pb['500M+']['recall']:.1%}) |"
        )

    md += [
        "",
        "## Severe Storm RF promotions",
        "",
        "| Disaster | Actual | Base | Specialist | Base correct | Specialist correct |",
        "|---:|---|---|---|---:|---:|",
    ]
    for _, r in changed_eval.iterrows():
        md.append(
            f"| {int(r['disasterNumber'])} | {r['actual_band_storm']} | {r['base_pred']} | "
            f"{r['final_pred']} | {bool(r['base_correct'])} | {bool(r['storm_correct'])} |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
