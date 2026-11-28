"""Synthetic-only check of counterbalancing and record validation. No data collection."""
import json
from collections import Counter

SCENARIOS = {tag: {r:f"{tag}-{r}" for r in ("normal","delayed","closed")} for tag in ("A","B")}
SEQUENCES = (
    (("unaided","A"),("assisted","B")),
    (("unaided","B"),("assisted","A")),
    (("assisted","A"),("unaided","B")),
    (("assisted","B"),("unaided","A")),
)

def allocation_failure(case, plan):
    q = plan.get("units")
    if not isinstance(q,list) or len(q)!=len(case["need"]):
        return "invalid_shape"
    if any(type(x) is not int or x<0 for x in q):
        return "invalid_quantity"
    if sum(q)>case["stock"]:
        return "stock"
    if any(v>cap for v,cap in zip(q,case["capacity"])):
        return "route"
    return None

def score(case, plan):
    failure=allocation_failure(case,plan)
    if failure:
        return {"feasible":False,"failure":failure,"primary":0}
    # Positive-need inaccessible zones remain in the denominator.
    fractions=[min(x,n)/n for x,n in zip(plan["units"],case["need"]) if n>0]
    return {"feasible":True,"failure":None,"primary":min(fractions)}

def review_record(condition, initial, recommendation, final, status):
    if condition=="unaided":
        assert recommendation is None and status is None and initial==final
    elif condition=="assisted":
        assert status in {"accept","edit","reject"} and isinstance(recommendation,dict)
        assert (final==recommendation)==(status=="accept")
        if status=="reject":
            assert initial==final
        if status=="edit":
            assert final!=initial
    else:
        raise ValueError("unknown condition")
    return dict(condition=condition,initial=initial,recommendation=recommendation,final=final,status=status)

def main():
    by_scenario=Counter()
    by_period=Counter()
    for seq in SEQUENCES:
        seen=set()
        for i,(condition,pack) in enumerate(seq,1):
            by_period[i,condition]+=1
            for scenario in SCENARIOS[pack].values():
                assert scenario not in seen
                seen.add(scenario)
                by_scenario[scenario,condition]+=1
    assert set(by_scenario.values())=={2} and set(by_period.values())=={2}
    demo={"need":[4,3,2],"stock":6,"capacity":[4,3,0]}
    initial={"units":[3,3,0]}
    recommendation={"units":[4,2,0]}
    assert score(demo,initial)=={"feasible":True,"failure":None,"primary":0}
    assert score(demo,{"units":[3,2,1]})["failure"]=="route"
    assert score(demo,{"units":[4,3,0]})["failure"]=="stock"
    assert score(demo,{"units":[4,2,-1]})["failure"]=="invalid_quantity"
    review_record("unaided",initial,None,initial,None)
    review_record("assisted",initial,recommendation,recommendation,"accept")
    review_record("assisted",initial,recommendation,initial,"reject")
    review_record("assisted",initial,recommendation,{"units":[2,3,0]},"edit")
    try:
        review_record("assisted",initial,recommendation,initial,"accept")
    except AssertionError:
        pass
    else:
        raise AssertionError("acceptance mismatch not rejected")
    print(json.dumps({"status":"simulation_only","participants":0,"schedule_slots":4,
                      "scenario_ids":6,"balanced_exposures_per_scenario_condition":2,
                      "negative_tests":"invalid route, stock, quantity and acceptance rejected",
                      "closed_zone_retained":True},sort_keys=True,indent=2))

if __name__=="__main__":
    main()
