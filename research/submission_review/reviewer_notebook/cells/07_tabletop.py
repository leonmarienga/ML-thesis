require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
tabletop = json.loads(evidence["tabletop_dry.json"])
require(tabletop["participants"] == 0 and tabletop["status"] == "simulation_only", "Tabletop scope drift")
require((tabletop["schedule_slots"], tabletop["scenario_ids"],
         tabletop["balanced_exposures_per_scenario_condition"], tabletop["closed_zone_retained"]) ==
        (4,6,2,True), "Archived tabletop invariant mismatch")
claims["tabletop"] = {"scope": "archived schedule dry run, not research with people", **tabletop}
print("Tabletop record: four schedule slots, six scenario IDs, zero participants.")
