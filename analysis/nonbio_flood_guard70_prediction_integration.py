#!/usr/bin/env python3
"""
Prediction-level integration audit for accepted 812 + strict Flood guard70.

Inputs are artifacts from two already-completed strict LFYO workflows:
1. accepted 812 full-router predictions
2. strict one-way Flood guard70 low-only predictions

No model is retrained and no threshold is reselected here.

Safety checks:
- only rows where strict guard70 changed base $100K-$1M -> $1M-$50M are applied;
- every changed row must be Flood;
- accepted 812 prediction must exactly equal the strict Flood audit's base
  prediction for every changed row;
- no row with actual funding >=$50M may be modified;
- high-band predictions remain bit-for-bit identical.
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "audit_inputs" / "flood_guard70_integration"
OUT = ROOT / "audit_outputs" / "flood_guard70_prediction_integration"
OUT.mkdir(parents=True, exist_ok=True)

ACCEPTED = IN / "accepted" / "plus_life_count_predictions.csv"
FLOOD = IN / "flood" / "rescue_guard70_predictions.csv"

BANDS = [
    "0-100K",
    "100K-1M",
    "1M-50M",
    "50-200M",
    "200-500M",
    "500M+",
]


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
    f = pd.read_csv(FLOOD)

    a["disasterNumber"] = a["disasterNumber"].astype(int)
    f["disasterNumber"] = f["disasterNumber"].astype(int)

    changed = f[
        (f["base_pred"] == "100K-1M")
        & (f["final_pred"] == "1M-50M")
    ].copy()

    assert len(changed) > 0, "Strict guard70 made no promotions."
    assert set(changed["incidentType"]) == {"Flood"}, (
        "Non-Flood row found among strict guard70 promotions."
    )

    cols = [
        "disasterNumber",
        "incidentType",
        "actual_band",
        "base_pred",
        "final_pred",
    ]
    changed = changed[cols].copy()

    merged = changed.merge(
        a[
            [
                "disasterNumber",
                "actual_band",
                "final_pred",
                "root_actual_high",
                "root_pred_high",
            ]
        ],
        on="disasterNumber",
        how="left",
        suffixes=("_flood", "_accepted"),
        validate="one_to_one",
    )

    assert merged["final_pred_accepted"].notna().all(), (
        "At least one strict Flood promotion is absent from accepted 812 predictions."
    )

    mismatches = merged[
        merged["final_pred_accepted"] != merged["base_pred"]
    ].copy()
    assert mismatches.empty, (
        "Accepted 812 base prediction disagrees with strict Flood audit for "
        f"{len(mismatches)} promoted rows: "
        f"{mismatches[['disasterNumber','base_pred','final_pred_accepted']].to_dict(orient='records')}"
    )

    assert (merged["root_actual_high"] == 0).all(), (
        "A strict Flood promotion touches an actual >=$50M row."
    )
    assert (merged["root_pred_high"] == 0).all(), (
        "A strict Flood promotion touches a row routed into the high branch."
    )

    integrated = a.copy()
    integrated["integrated_pred"] = integrated["final_pred"]

    rescue_map = dict(
        zip(
            changed["disasterNumber"],
            changed["final_pred"],
        )
    )
    mask = integrated["disasterNumber"].isin(rescue_map)
    integrated.loc[mask, "integrated_pred"] = (
        integrated.loc[mask, "disasterNumber"].map(rescue_map)
    )

    high_actual = integrated["root_actual_high"] == 1
    assert (
        integrated.loc[high_actual, "integrated_pred"]
        == integrated.loc[high_actual, "final_pred"]
    ).all()

    base = metrics(integrated, "final_pred")
    new = metrics(integrated, "integrated_pred")

    changed_eval = merged[
        [
            "disasterNumber",
            "incidentType",
            "actual_band_flood",
            "base_pred",
            "final_pred_flood",
            "final_pred_accepted",
        ]
    ].copy()
    changed_eval["base_correct"] = (
        changed_eval["base_pred"] == changed_eval["actual_band_flood"]
    )
    changed_eval["rescued_correct"] = (
        changed_eval["final_pred_flood"] == changed_eval["actual_band_flood"]
    )

    integrated.to_csv(OUT / "integrated_predictions.csv", index=False)
    changed_eval.to_csv(OUT / "guard70_promotions.csv", index=False)

    summary = {
        "source_runs": {
            "accepted_812": 34339229661,
            "strict_flood_guard70": 34360664410,
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
        "# Accepted 812 + strict Flood guard70 prediction integration",
        "",
        "- Uses only completed strict GitHub workflow predictions.",
        "- No model retraining or threshold reselection.",
        "- All strict guard70 promotions matched the accepted 812 low prediction exactly.",
        "- No actual or routed >=$50M row was modified.",
        "",
        "| Variant | Overall | Macro | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, m in [("accepted_812", base), ("plus_flood_guard70", new)]:
        pb = m["per_band"]
        md.append(
            f"| {name} | "
            f"{m['overall_correct']}/{m['overall_total']} ({m['overall_accuracy']:.1%}) | "
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
        "## Guard70 promotions",
        "",
        "| Disaster | Actual | Base | Guard70 | Base correct | Guard70 correct |",
        "|---:|---|---|---|---:|---:|",
    ]
    for _, r in changed_eval.iterrows():
        md.append(
            f"| {int(r['disasterNumber'])} | {r['actual_band_flood']} | "
            f"{r['base_pred']} | {r['final_pred_flood']} | "
            f"{bool(r['base_correct'])} | {bool(r['rescued_correct'])} |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
