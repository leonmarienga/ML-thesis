#!/usr/bin/env python3
"""Independent USACE flood loss demonstration without administrative outcome labels."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


def aggregate(raw):
    doc=json.loads(raw.read_text())
    assert doc["type"]=="FeatureCollection"
    stocks=[]
    for item in doc["features"]:
        p=item["properties"]
        assert str(p["cbfips"]).startswith("22063")
        if p["occtype"]=="RES1-1SNB" and p["val_struct"] is not None and p["val_struct"]>0:
            stocks.append(float(p["val_struct"]))
    assert stocks
    return {"county_fips":"22063","county_name":"Livingston Parish, Louisiana",
            "occupancy":"RES1-1SNB","all_inventory_structures":len(doc["features"]),
            "eligible_structures":len(stocks),"inventory_replacement_value_usd":sum(stocks),
            "nsi_public_payload_sha256":sha256(raw.read_bytes()).hexdigest(),
            "underlying_inventory_is_modeled":True}


def study(curve, inventory):
    result=[]
    for share in (.01,.05,.20):
        value=inventory["inventory_replacement_value_usd"]*share
        for record in curve[curve.depth_ft.isin([0,1,2,4,6,8,10])].itertuples(index=False):
            structural=value*record.structure_fraction_of_structure_value
            contents=value*record.contents_fraction_of_structure_value
            result.append({"scenario_id":f"EGM0103-{share:.2f}-{int(record.depth_ft):02d}",
                "scenario_exposed_share":share,"assumed_depth_ft_above_first_floor":record.depth_ft,
                "estimated_exposed_replacement_value_usd":value,
                "simulated_structure_loss_usd":structural,"simulated_contents_loss_usd":contents,
                "simulated_direct_loss_usd":structural+contents,
                "humanitarian_needs_status":"unmeasured",
                "fema_mission_assignment_obligations_status":"unassigned",
                "eligible_for_obligation_model_training":False,
                "is_historical_observation":False})
    return pd.DataFrame(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    input_=p.add_mutually_exclusive_group(required=True)
    input_.add_argument("--inventory-api-file",type=Path)
    input_.add_argument("--retained-inventory-summary",type=Path)
    p.add_argument("--damage-curve",type=Path,default=Path("research/augmentation/inputs/usace_egm0103_flood_curve.csv"))
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    inventory=aggregate(args.inventory_api_file) if args.inventory_api_file else json.loads(args.retained_inventory_summary.read_text())
    curve=pd.read_csv(args.damage_curve)
    assert curve.depth_ft.is_unique and curve.depth_ft.is_monotonic_increasing
    assert all(curve[c].is_monotonic_increasing and curve[c].between(0,1).all() for c in curve.columns[1:])
    d=study(curve,inventory)
    assert len(d)==21 and d.simulated_direct_loss_usd.ge(0).all()
    assert d.groupby("scenario_exposed_share").simulated_direct_loss_usd.apply(lambda x:x.is_monotonic_increasing).all()
    (args.output/"physics_inventory_summary.json").write_text(json.dumps(inventory,indent=2)+"\n")
    d.to_csv(args.output/"physics_impact_pilot.csv",index=False)


if __name__=="__main__":
    main()
