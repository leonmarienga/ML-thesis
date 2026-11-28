require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
fwd = rows("forward_t5.csv")
require([r["actual_band"] for r in fwd] == bands, "Forward matrix bands changed")
n = [int(r["actual_n"]) for r in fwd]
require(sum(n) == 376 and min(n) == 1, "Unexpected matched case support")
forward = {}
for label in ("retrospective_metadata", "verified_link"):
    d = [int(r[label+"_hits"]) for r in fwd]
    recall = [k/t for k, t in zip(d, n)]
    require(all(near(p, float(r[label+"_recall"])) for p, r in zip(recall, fwd)),
            "Archived per-class forward recall mismatch")
    forward[label] = {"correct":sum(d), "accuracy_pct":round(100*sum(d)/sum(n), 2),
                      "balanced_accuracy_pct":round(100*mean(recall), 2)}
require(forward["verified_link"]["accuracy_pct"] > forward["retrospective_metadata"]["accuracy_pct"] and
        forward["verified_link"]["balanced_accuracy_pct"] < forward["retrospective_metadata"]["balanced_accuracy_pct"],
        "Forward trade-off changed")
claims["matched_forward"] = {"scope":"same retrospective held cases, not an independent prospective release",
                             "n":sum(n),"band_support":n, "metrics": forward}
print("Same 376 matched cases; raw and class-balanced accuracy (%):")
for label, scores in forward.items():
    print(f"{label}: {scores['accuracy_pct']:.2f}, {scores['balanced_accuracy_pct']:.2f}")
matrix = rows("forward_matrices.csv")
variants = {"retrospective_metadata":"ablation_retrospective_metadata", "verified_link":"ablation_verified_link_only"}
for label, variant in variants.items():
    counts = {(r["actual_band"], r["predicted_band"]): int(r["n"]) for r in matrix if r["variant"] == variant}
    require(set(counts) == {(x,y) for x in bands for y in bands}, "Incomplete forward confusion matrix")
    require([sum(counts[b,p] for p in bands) for b in bands] == n and
            [counts[b,b] for b in bands] == [int(r[label+"_hits"]) for r in fwd],
            "T5 and source forward matrices disagree")
compare = one(rows("forward_bootstrap.csv"), variant="ablation_verified_link_only")
mae = float(compare["mae_difference_usd"])
mae_ci = [float(compare[k]) for k in ("mae_ci_low", "mae_ci_high")]
require(mae_ci[0] < 0 < mae < mae_ci[1] and
        near(float(compare["accuracy_difference"]), (224-192)/376),
        "Archived bootstrap contrast changed")
claims["matched_forward"].update(archived_mae_change_usd=mae, archived_mae_ci95_usd=mae_ci)
print(f"Archived episode-bootstrap MAE change ${mae:,.0f}: its interval crosses zero.")
