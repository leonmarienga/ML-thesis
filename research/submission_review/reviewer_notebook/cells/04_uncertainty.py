require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
intervals = rows("interval_t8.csv")
cts = ["n","covered","above_upper","below_lower"]
interval_report = {}
require(len(intervals) == 12, "Unexpected interval rows")
for level in ("0.80000000", "0.90000000"):
    for pipeline in ("residual", "cqr"):
        matched = [one(intervals, nominal=level, pipeline=pipeline, scope=s)
                   for s in ("all", "under_50m", "50m_or_more")]
        counts = [{k:int(r[k]) for k in cts} for r in matched]
        require(all(c["n"] == c["covered"]+c["above_upper"]+c["below_lower"] for c in counts),
                "Interval hit/miss counts do not reconcile")
        require(all(counts[0][k] == counts[1][k]+counts[2][k] for k in cts) and
                [c["n"] for c in counts] == [308,299,9], "Interval strata do not aggregate")
        label = f"{float(level):.0%} {pipeline}"
        interval_report[label] = {r["scope"]:{**c,"coverage_pct":round(100*c["covered"]/c["n"],2),
                                             "median_width_usd":float(r["median_width_usd"])}
                                  for r,c in zip(matched,counts)}
        print(f"{label}: overall {counts[0]['covered']}/308, >=$50M {counts[2]['covered']}/9")
require(interval_report["90% residual"]["all"]["covered"] == 281 and
        interval_report["90% residual"]["50m_or_more"]["covered"] == 0, "90% residual finding changed")
claims["intervals"] = {"scope":"archived matched aggregate; very small tail, no out-of-sample guarantee",
                       "results":interval_report}
print("A nominal level is not a tail guarantee; CQR tail breadth and support matter.")
audit = rows("uncertainty_review.csv")
for scope, key in (("all","all"), ("below_50m","under_50m"), ("at_least_50m","50m_or_more")):
    original = interval_report["90% residual"][key]
    independent_aggregate = one(audit, scope=scope)
    require(all(int(independent_aggregate[k]) == original[k] for k in ("n","covered")),
            "90% interval aggregate did not reconcile to audit table")
require(int(one(audit,scope="all")["flagged_misses"]) == 23 and
        int(one(audit,scope="at_least_50m")["misses"]) == 9, "Retrospective flag counts changed")
