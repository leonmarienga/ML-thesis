require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
metrics = ("overall","high_vulnerability","least_served")
archived_floors = rows("transport_floors.csv")
floor = {}
for metric in metrics:
    f = one(archived_floors, policy="vulnerability", metric=metric)
    delta = float(f["difference"])
    lo, hi = float(f["lower_95"]), float(f["upper_95"])
    require(lo <= delta <= hi, "Invalid archived floor range")
    floor[metric] = {"change_pp":100*delta, "archived_interval_pp":[100*lo,100*hi]}
require(floor["least_served"]["change_pp"] > 0 > floor["high_vulnerability"]["change_pp"],
        "Floor policy trade-off changed")
print("Batch 14 synthetic geographic floor (0 to 100% of reference), differences in percentage points:")
for k, v in floor.items():
    print(f"  {k}: {v['change_pp']:+.2f}")
claims["transport_floor"] = {"scope":"archived simulator contrasts, not observed service", "metrics":floor}
pairs = rows("veto_contrasts.csv")
arm = "optimizer_with_scripted_veto"
require(len(pairs) == 9, "Unexpected veto aggregate shape")
for r in pairs:
    require(sum(int(r[k]) for k in ("positive","tied","negative")) == int(r["n"]) == 48,
            "Veto contrast matched support invalid")
    require(near(float(r["left_mean"])-float(r["right_mean"]), float(r["difference"])),
            "Veto contrast means do not reconcile")
veto = {}
for metric in metrics:
    r = one(pairs, left=arm, right="optimizer_only", metric=metric)
    delta = float(r["left_mean"])-float(r["right_mean"])
    lo, hi = float(r["lower_95"]), float(r["upper_95"])
    require(lo <= delta <= hi, "Invalid archived veto range")
    veto[metric] = {"change_pp":100*delta, "archived_interval_pp":[100*lo,100*hi]}
require(veto["least_served"]["change_pp"] > 0 > veto["overall"]["change_pp"] and
        veto["high_vulnerability"]["change_pp"] < 0, "Veto trade-off changed")
print("Separate 48-world synthetic optimizer+scripted veto versus optimizer, percentage points:")
for k, v in veto.items():
    print(f"  {k}: {v['change_pp']:+.2f}")
claims["policy_veto"] = {"scope":"synthetic scripted governance, not participants or operational effect", "n":48, "metrics":veto}
