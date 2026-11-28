require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
historic = rows("track_a_bands.csv")
bands = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
require([r["actual_band"] for r in historic] == bands, "Track A bands changed")
totals = [sum(int(r[b]) for b in bands) for r in historic]
hits = [int(r[b]) for r, b in zip(historic, bands)]
require(totals == [int(r["actual_n"]) for r in historic] and (sum(hits), sum(totals)) == (830, 912),
        "Track A confusion counts mismatch")
require(all(near(100*k/n, float(r["recall_pct"]), 1e-7) for k, n, r in zip(hits, totals, historic)),
        "Track A recalls mismatch")
versions = {r["amount_version"]: float(r["mae_usd"]) for r in rows("track_a_amounts.csv")}
require(len(versions) == 3 and near(versions["challenger_stack_prediction"], 5014819.011614494, 0.01),
        "Track A amount version mismatch")
claims["track_a"] = {"scope":"archived reconstruction; target-conditioned rules and non-forward LFYO",
                     "correct":sum(hits), "total":sum(totals), "accuracy_pct":round(100*sum(hits)/sum(totals), 2),
                     "band_totals":totals, "band_hits":hits, "archived_amount_mae_usd":versions}
print(f"Track A archive: {sum(hits)}/{sum(totals)} = {100*sum(hits)/sum(totals):.2f}%; not forward skill.")
for key, usd in versions.items():
    print(f"Reported {key}: MAE ${usd:,.2f}; read from the pinned aggregate, not re-scored.")
