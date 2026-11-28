#!/usr/bin/env python3
"""Extract selected evidence from immutable aggregate inputs; no model execution."""
import argparse, csv, hashlib, io, json, subprocess
from pathlib import Path

P="b0b6edb229d112dd6d11896a512bf1112414e4f6"
REV="1ca331472ae68fd75a5570bc9270ee877fb7a61f"
G="https://github.com/leonmarienga/ML-thesis"
C="research/continuous/"
A="research/augmentation/"
W=C+"weight_influence/results/"
TA="research/results/track_a/model_metrics.csv"
FW="research/results/forward/model_metrics.csv"
p=argparse.ArgumentParser()
p.add_argument("repo",type=Path)
p.add_argument("out",type=Path)
arg=p.parse_args()
subprocess.check_output(["git","-C",str(arg.repo),"cat-file","-e",P+"^{commit}"])
arg.out.mkdir(parents=True,exist_ok=True)
inputs={}

def link(path,ref=P): return f"{G}/blob/{ref}/{path}"
def record(path,ref,obj):
    oid=hashlib.sha1(f"blob {len(obj)}\0".encode()+obj).hexdigest()
    inputs[(ref,path)]=dict(commit=ref,path=path,git_blob=oid,sha256=hashlib.sha256(obj).hexdigest())
    return oid
def get(path):
    # Only hardcoded historical aggregate, code or reporting files below are read.
    assert not any(x in path.casefold() for x in ("fy2025","predictions.csv","historical_actions","synthetic.csv"))
    obj=subprocess.check_output(["git","-C",str(arg.repo),"show",f"{P}:{path}"])
    record(path,P,obj)
    return obj
def read(path): return list(csv.DictReader(io.StringIO(get(path).decode())))
def one(rows,**kv):
    z=[r for r in rows if all(r[k]==str(v) for k,v in kv.items())]
    assert len(z)==1,kv
    return z[0]
def write(file,rows):
    with (arg.out/file).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader(); w.writerows(rows)
def money(x): return f"USD {float(x)/1e6:.2f}M"
def d(x,n=4): return f"{float(x):.{n}f}"
def action(num): return f"{G}/actions/runs/{num}"

loc=Path(__file__).resolve().parent/"sources"
later={}
for u in json.loads((loc/"FOLLOWUP_MANIFEST.json").read_text()):
    b=(loc/u["file"]).read_bytes()
    assert record(u["path"],u["ref"],b)==u["sha"]
    later[u["file"]]=(json.loads(b),link(u["path"],u["ref"]),u["ref"])

models=[]
def model(key,stage,endpoint,n,unit,method,finding,bound,src,res,m=None,acc=None,corr=None,replay=""):
    models.append(dict(model_id=key,disposition=stage,endpoint=endpoint,scored_n=n,score_unit=unit,validation_method=method,
                       routes_correct=corr,accuracy_fraction=acc,mae_usd=m,manuscript_result=finding,evidence_boundary=bound,
                       method_url=src,evidence_url=res,replay_or_crosscheck_url=replay))
at=read(TA); registry=read("research/MODEL_REGISTRY.csv")
lg="Retrospective leave-fiscal-year-out archived replay"
caveat="Four late rules used the observed obligation to filter held-row eligibility; not a target-blind or operational score."
cross=link("research/submission_review/SIX_BAND_METHODOLOGY.md",REV)
router="analysis/nonbio_829_hurricane_amendment_floor.py"
for script in (router,"analysis/nonbio_825_severe_ice_operational_dominance.py",
               "analysis/nonbio_826_fire_duration_floor.py","analysis/nonbio_827_hurricane_zero_mission_prereq.py"):
    assert "target_clean" in get(script).decode() and "<50_000_000" in "".join(get(script).decode().split())
model("track_a_router_830","immutable_historical_router","Six bands",912,"declarations",lg,
      "830/912 archived routes correct (91.01%); same fixed routes used by all three dollar treatments.",caveat,
      link(router),cross,acc=.9100877192982456,corr=830,replay=action(34701792537))
for key,stage,summary,method,replay in [
    ("track_a_original_stack","frozen_original_amount_stack","Original medians in bands 1, 2 and 4, ExtraTrees in band 3, separate upper specialists.",
     "analysis/nonbio_final_830_accepted_amount_stack.py",action(34796746603)),
    ("track_a_two_accepted_low_specialists","accepted_historical_amount_overlay","Only two lowest-band amount estimators replaced; predicted routes unchanged.",
     "analysis/nonbio_final_50_200m_immutable_overlay.py",cross),
    ("track_a_reported_historical_50_200m_overlay","reported_research_only_amount_overlay","Later overlay changes only routed 50–<200M amounts relative to accepted low-band stack; not an accepted replacement.",
     "analysis/nonbio_final_50_200m_immutable_overlay.py",action(41397678781)),
]:
    get(method)
    a=one(at,variant=key,axis="all",group="all"); reg=one(registry,model_id=key)
    assert a["n"]=="912" and float(a["accuracy"])*912==830 and a["mae_usd"]==reg["mae_usd"]
    model(key,stage,"Historical dollar amounts",912,"declarations",lg,
          f"USD {float(a['mae_usd'])/1e6:.3f}M all-case MAE. {summary}",caveat,link(method),link(TA),
          m=a["mae_usd"],acc=a["accuracy"],corr=830,replay=replay)
f=read(FW)
for key,stage in [("ablation_retrospective_metadata","retrospective_forward_reference"),("ablation_verified_link_only","research_only_forward_challenger")]:
    a=one(f,variant=key,axis="all",group="all"); n=int(a["n"]); assert n==376
    q=float(a["accuracy"])
    model(key,stage,"Six bands and dollar amounts",n,"declarations","Matched FY2019–24 date/episode-purged forward",
          f"{100*q:.2f}% exact bands, {money(a['mae_usd'])} MAE. See T5 for paired changes.",
          "Retrospective features and repeated historical test population; not a ready decision-time product.",
          link("analysis/thesis_forward_models.py"),link(FW),m=a["mae_usd"],acc=q,corr=round(q*n))
tb,tb_url,tb_ref=later["target_blind_summary.json"]; b=tb["baseline"]; q=tb["candidate"]
assert (b["correct"],q["correct"],q["n"],q["episode_groups"])==(273,274,376,325)
assert tb["held_outcome_stress_origins_passed"]==6 and not tb["independent_replication_priority_gate_met"] and not tb["fy_2025_accessed"]
model("target_blind_router_v1","research_only_no_promotion","Six bands",376,"declarations","Metadata-only chronological six-origin, episode-purged trial",
      "273/376 fresh baseline, 274/376 candidate; exact held-label perturbation invariance across all six origins.",
      "Three changes in two groups; paired macro-F1 interval includes zero; not a counterfactual Track A score.",
      link("research/followup_20261128/target_blind_router_v1/PROTOCOL.md",tb_ref),tb_url,acc=q["accuracy"],corr=q["correct"])
w=read(W+"metrics.csv"); wr=one(w,rank=0,arm="real_only"); wh=one(w,rank=0,arm="weighted_20x")
assert (wh["n"],wh["tail_n"],wh["tail_groups"])==("135","11","10")
model("batch13_signed_weight_fixed20","research_only_fixed_candidate_preserved","Signed 36-month obligations",135,"declarations",
      "Matched FY2019–23 elapsed cohort; earlier-only sourcing, maturity and episode gate",
      f"{money(wh['mae'])} vs {money(wr['mae'])} overall MAE; {money(wh['tail_mae'])} vs {money(wr['tail_mae'])} tail MAE.",
      "Previously examined outcome population; no calibrated threshold accepted or independent comparable validation.",
      link("analysis/thesis_weight_influence.py"),link(W+"metrics.csv"),m=wh["mae"])
for key,fn,description in [
    ("forward_residual_dollar_and_log1p","thesis_uncertainty.py","Calibrates on earlier forward predictions and excludes recurrent held groups."),
    ("archived_track_b_cqr","cqr_track_b_fixed_split.py","Fit through FY2018, calibrate FY2019, test FY2020–24; not episode-purged."),
]:
    reg=one(registry,model_id=key); assert reg["validation_rows"]=="308"
    model(key,"research_only_interval_comparator","Historical coverage and width",308,"declarations",
          "Matched FY2020–24 scoring in T8",description,
          "Retrospective intervals do not imply chronological or tail/group coverage guarantees.",
          link("analysis/"+fn),link("research/submission_review/figures/uncertainty/T8-matched-intervals.csv",REV))
notebook=json.loads(get("reinforcement_learning.ipynb"))
outs="".join(json.dumps(c["outputs"]) for c in notebook["cells"] if c.get("cell_type")=="code")
assert "191.87" in outs and "319.58" in outs
model("legacy_clean_dqn_vs_ml_only","archived_budget_simulation_only","Historical obligation-referenced reward",
      30,"simulated episodes","Seeded episodes from the same dataset used in fit/tune",
      "Mean reward −191.87 DQN vs −319.58 ML-only; dollar MAE worsened (15.44M vs 14.30M USD).",
      "Hard released-budget cap applies to all policies; no independent disasters or measured relief outcomes.",
      link("reinforcement_learning.ipynb"),cross)
model("batch14_transport_multi_objective","simulation_only","Simulated logistics service",48,"simulated worlds",
      "Paired held synthetic worlds, see T9","Capacity/stock/arrival-yield LP and recourse.",
      "No real deployment, vehicle routing or measured humanitarian effects.",
      link("analysis/thesis_transport_pareto.py"),link(C+"transport_pareto/results/preference_choices.csv"))
assert len({m["model_id"] for m in models})==len(models)
write("T3-model-lineage.csv",models)

gates=[]; metrics=[]
def gate(key,study,scope,status,result,bound,src,support=""):
    gates.append(dict(gate_id=key,study=study,population_and_unit=scope,disposition=status,manuscript_result=result,
                      evidence_boundary=bound,evidence_url=src,support_url=support))
def metric(key,measure,value,unit,arm,src,lo=None,hi=None):
    if lo is not None: assert hi is not None and float(lo)<=float(hi)
    metrics.append(dict(gate_id=key,metric=measure,value=value,unit=unit,comparison=arm,
                        interval_95_lower=lo,interval_95_upper=hi,evidence_url=src))
AR=A+"results/"
su=read(AR+"support.csv")
max_donors=max(int(x["max_episodes_per_state_hazard"]) for x in su)
assert max_donors==2
assert all(x["state_hazard_strata_ctgan_gate"]==x["state_hazard_strata_copula_gate"]=="0" for x in su)
ga=A+"GENERATOR_AUDIT.md"
assert all(s in get(ga).decode() for s in ("required eight independent episodes","required 100 independent compatible training episodes","98 of 359 draws"))
gate("generator_support","Monetary generative support","Earlier training compartments; compatible state × hazard × band",
     "blocked_no_generator_fit","At most 2 earlier compatible episodes. This study prespecified minima of 8 for the copula and 100 for CTGAN.",
     "Protocol-specific gates; no trained copula or CTGAN result.",link(AR+"support.csv"),link(ga))
for k,v in [("maximum_compatible",max_donors),("copula_minimum",8),("ctgan_minimum",100)]:
    metric("generator_support",k,v,"episodes","Within compatible strata",link(ga))
ar=read(AR+"metrics.csv")
ar={x["variant"]:x for x in ar if x["probability_scope"]=="calibrated" and x["variant"].startswith("nested_")}
ic=[r for r in read(AR+"intervals.csv") if r["subset"]=="at_least_50m"]
assert {x["n"] for x in ic}=={"13"} and {x["level"] for x in ic}=={"80","90"} and {x["covered"] for x in ic}=={"0"}
assert {x["n"] for x in ar.values()}=={"376"} and {x["historical_tail_episode_groups"] for x in ar.values()}=={"11"}
gate("monetary_interpolation","SMOTE-family and downweighted hybrid","376 real FY2019–24 declarations; 13 ≥50M in 11 groups",
     "evaluated_no_material_novel_gain","0/13 tail cases covered at nominal 80% and 90% in every configuration; no novel generator improved the joint endpoints.",
     "Inactive nested SMOTE/copula counterparts are real-only fallbacks. SMOGN repeated a real feature vector in 98/359 draws while perturbing amount.",
     link(AR+"intervals.csv"),link(ga))
for v in ("nested_real","nested_smotenc","nested_smoter","nested_smogn","nested_hybrid"):
    for a,u in [("tail_mae_usd","USD"),("tail_average_precision","fraction")]:
        metric("monetary_interpolation",a,ar[v][a],u,v,link(AR+"metrics.csv"))
for h in (80,90):
    metric("monetary_interpolation","tail_covered",0,"declarations",f"Nominal {h}%; 13 observed tail cases",link(AR+"intervals.csv"))
gate("monetary_weight_repeat","Training weights and repeated examples (controls)","Same 376 held workbook declarations",
     "evaluated_controls_only","Tail MAE: real-only 553.27M USD, weights 500.60M, repeats 540.61M; weighting worsened overall MAE.",
     "Not novel samples, not the signed fixed 20× experiment, and not independent.",link(AR+"metrics.csv"),link(AR+"paired_episode_bootstrap.csv"))
for v in ("nested_real","nested_weight","nested_random"):
    for a in ("mae_usd","tail_mae_usd"):
        metric("monetary_weight_repeat",a,ar[v][a],"USD",v,link(AR+"metrics.csv"))
py=read(AR+"physics_impact_pilot.csv")
assert len(py)==21 and all(row["fema_mission_assignment_obligations_status"]=="unassigned"
    and row["eligible_for_obligation_model_training"]=="False"
    and row["is_historical_observation"]=="False" for row in py)
gate("usace_damage_not_labels","USACE synthetic flood property-loss pilot","21 modeled depth/exposure scenarios; no measured footprint",
     "feasibility_only_no_funding_bridge","All 21 modeled scenarios explicitly mark obligations unassigned; none is a training label.",
     "Modeled direct economic loss is not an actual obligation or humanitarian need.",link(AR+"physics_impact_pilot.csv"),link(ga))
metric("usace_damage_not_labels","scenarios_without_obligation_labels",21,"scenarios","All modeled",link(AR+"physics_impact_pilot.csv"))
wb=one(read(W+"original_bootstrap.csv"),endpoint="signed_elapsed",arm="weighted_20x",metric="tail_mae")
assert float(wb["lower_95"])<0<float(wb["upper_95"])
gate("signed_fixed_20x","Fixed 20× signed 36-month obligations (Batch 13)","135 matched historical declarations; 11 ≥50M in 10 groups",
     "preserve_exploratory_no_promotion",
     "Tail MAE 347.66M → 291.80M USD, Δ −55.86M (95% CI −143.28M to +2.50M); 11/11 detections and 39 false alerts vs 8/11 and 13.",
     "Uncalibrated 0.10 alert. Requires new independent comparable signed labels; neither risk policy nor amount improvement established.",
     link(W+"metrics.csv"),link(W+"original_bootstrap.csv"))
for m in (wr,wh):
    for a,u in [("mae","USD"),("tail_mae","USD"),("tail_brier","fraction squared"),("alerts_true_10pct","cases"),("alerts_false_10pct","cases")]:
        metric("signed_fixed_20x",a,m[a],u,m["arm"],link(W+"metrics.csv"))
metric("signed_fixed_20x","paired_tail_mae_delta",wb["change"],"USD","Weighted minus real-only",
       link(W+"original_bootstrap.csv"),wb["lower_95"],wb["upper_95"])
wo=read(W+"paired_differences.csv")
assert float(one(wo,rank=1)["mae"])>0 and all(float(x["tail_mae"])<0 for x in wo)
gate("signed_20x_influence","Fixed 20× episode influence","Same held signed declarations; earlier source groups omitted by rank",
     "evaluated_diagnostic_not_independent","Removing the largest earlier group leaves 21.62M USD tail gain but reverses overall change to 0.45M worse.",
     "Training-only omission is sensitivity analysis, not a new test population.",link(W+"paired_differences.csv"),link(W+"omitted_groups.csv"))
for m in wo:
    for k in ("mae","tail_mae"):
        metric("signed_20x_influence","paired_"+k+"_delta",m[k],"USD","Omission rank "+m["rank"],link(W+"paired_differences.csv"))

s0=C+"coverage_specialists/results/"
s=read(s0+"scores.csv"); hh=one(s,endpoint="signed_elapsed",population="pooled",arm="hazard_gate_10")
assert hh["active_cases"]=="57"
gate("signed_other_specialists","Fixed hazard gate, positive hurdle and nested selector (Batch 11)","Same 135 prior-inspected signed declarations; fixed 0.10 hazard gate active on 57",
     "evaluated_research_only","Fixed hazard gate 310.80M USD tail MAE with real-only fallback; nested selector 345.84M vs 347.66M real-only.",
     "Conditional activation and fallback matter; no independent corpus or promotion.",link(s0+"scores.csv"),link(s0+"paired_changes.csv"))
for arm in ("real_only","positive_hurdle","hazard_gate_10","nested_selected"):
    m=one(s,endpoint="signed_elapsed",population="pooled",arm=arm)
    metric("signed_other_specialists","tail_mae",m["tail_mae"],"USD",arm,link(s0+"scores.csv"))

h=C+"horizons/results/past_only_support.csv"; le=C+"legacy_ledger_gate/results/gate_summary.csv"; dt=C+"documentary_extension/results/documentary_gate_summary.csv"
hp=[x for x in read(h) if x["policy"]=="elapsed_36m_plus_90d_at_origin" and x["threshold_usd"]=="500000000"]
assert len(hp)==6 and sorted(int(m["exceedance_episodes"]) for m in hp)==[0,0,0,2,2,2]
m=read(le)[0]
assert (m["legacy_declarations_checked"],m["legacy_actions"],m["eligible_new_elapsed_labels"])==("5","0","0")
q=one(read(dt),evidence_tier="combined_distinct_evidence_tiers",horizon_hours=72)
assert q["rare_eligible_origins"]==q["general_eligible_origins"]=="0"
gate("signed_independent_support","Signed action maturity, legacy ledger and decision-time documentary gates",
     "Six historical origins, five exact precoverage declarations, combined T+72 screen",
     "blocked_no_new_comparable_labels","0–2 prior ≥500M episodes per mature origin; zero eligible new legacy labels; zero eligible rare T+72 origins.",
     "Unpublished actions are unknown rather than zero; archive retrieval does not authenticate issuance.",link(h),link(le))
for m in hp: metric("signed_independent_support","prior_ge_500m",m["exceedance_episodes"],"episode groups","FY "+m["held_fy"],link(h))
metric("signed_independent_support","eligible_new_legacy",0,"labels","Five exact queries",link(le))
metric("signed_independent_support","eligible_rare_t72",0,"origins","Combined distinct evidence tiers",link(dt))

e=C+"earthquake/results/"
m=read(e+"pga_metrics.csv"); r=one(m,model="real_only"); a=one(m,model="hybrid_100"); b=one(read(e+"pga_paired_bootstrap.csv"),model="hybrid_100")
assert a["held_episodes"]==b["episodes"]=="7" and a["held_stations"]=="143"
gate("earthquake_bjf97","Physics-derived PGA hybrid","Seven later earthquakes, 143 instrumented stations",
     "auxiliary_below_gate",f"Equal-event log-PGA MAE {d(r['equal_episode_log_mae'])} → {d(a['equal_episode_log_mae'])}; episodic CI includes zero.",
     "Physical auxiliary endpoint; no new FEMA obligation or measured relief labels.",link(e+"pga_metrics.csv"),link(e+"pga_paired_bootstrap.csv"))
metric("earthquake_bjf97","paired_log_pga_mae_delta",b["difference_vs_real_only"],"log PGA","Hybrid 100 minus real Ridge",link(e+"pga_paired_bootstrap.csv"),b["p025"],b["p975"])

o=C+"disturbance_duration/results/"
m=read(o+"scores.csv")
r=one(m,population="all",model="empirical_seasonal"); a=one(m,population="all",model="hybrid_half_synthetic")
gain=1-float(a["brier_24h"])/float(r["brier_24h"])
b=one(read(o+"paired_brier_intervals.csv"),block="block14")
assert a["n"]=="172" and 0<gain<.05
gate("doe417_duration_prior","Seasonal simulated duration mixture (DOE-417)","172 held weather disturbance reports, 82 above 24 h",
     "auxiliary_below_practical_gate",f"24-hour Brier {d(r['brier_24h'],5)} → {d(a['brier_24h'],5)}; {100*gain:.2f}% relative gain vs required 5%.",
     "Below practical threshold despite favorable block intervals; historical timing and endpoint completion remain incomplete.",
     link(o+"scores.csv"),link(o+"paired_brier_intervals.csv"))
metric("doe417_duration_prior","relative_brier_gain",gain,"fraction","Hybrid versus seasonal",link(o+"scores.csv"))
metric("doe417_duration_prior","paired_brier_delta",b["hybrid_minus_empirical_brier"],"fraction squared","14-day blocks",
       link(o+"paired_brier_intervals.csv"),b["lower_95"],b["upper_95"])
o=C+"sustained_outage/results/"
m=read(o+"scores.csv")
r=one(m,population="all_held",model="real_logistic"); a=one(m,population="all_held",model="hybrid_logistic")
b=one(read(o+"paired_brier_intervals.csv"),block_days=14)
assert a["rows"]=="2139" and float(b["paired_95_lower"])<0<float(b["paired_95_upper"])
gate("eaglei_markov_transfer","Synthetic Markov county-outage trajectories","2,139 complete held county windows; not independent storms",
     "evaluated_no_gain",f"Brier {d(r['brier'],5)} → {d(a['brier'],5)}; paired interval includes zero.",
     "No patient/household restoration, need or funding model inference.",link(o+"scores.csv"),link(o+"paired_brier_intervals.csv"))
metric("eaglei_markov_transfer","paired_brier_delta",b["hybrid_minus_real_brier"],"fraction squared","14-day blocks",
       link(o+"paired_brier_intervals.csv"),b["paired_95_lower"],b["paired_95_upper"])
o=C+"storm_outage_panel/"
for v,n in [("results",1),("zero_sensitivity",0)]:
    path=o+v+"/gate_summary.csv"; m=one(read(path),population="new_storms")
    assert m["qualifying_held_storms"]==str(n) and m["gate_pass"]=="False"
    metric("wind_outage_storm_support","qualified_new_held",n,"storms",v,link(path))
gate("wind_outage_storm_support","Wind-conditioned county outage simulation readiness","Count qualifying held storms outside previously reviewed storm investigations",
     "blocked_no_generator_fit","1 new qualifying held storm (required 2); 0 after exclusion of explicit numeric zero counts.",
     "Windows are not extra independent groups. Zero-count convention unresolved.",
     link(o+"results/gate_summary.csv"),link(o+"zero_sensitivity/gate_summary.csv"))

o=C+"coastal_transfer/results/"
for v in ("verified","unflagged"):
    path=o+v+"/contrasts.csv"; m=read(path)
    for lead in (1,6):
        b=one(m,lead=lead,arm="hybrid",reference="statistical")
        assert b["held_storms"]=="4" and float(b["delta_m"])>0
        metric("coastal_slab_ridge_transfer","hybrid_minus_real_mae",b["delta_m"],"m",f"{v} {lead}h",link(path),b["lower_95"],b["upper_95"])
    a=one(m,lead=6,arm="statistical",reference="persistence")
    if v=="unflagged": assert float(a["delta_m"])>0
    else:
        assert float(a["delta_m"])<0
        rr=float(a["storm_mean_mae_m"]); ps=rr-float(a["delta_m"])
        hh=float(one(m,lead=6,arm="hybrid",reference="statistical")["storm_mean_mae_m"])
gate("coastal_slab_ridge_transfer","Synthetic pressure/wind prior and real ridge","Eight earlier, four held storms; 502 matched six-hour origins in primary",
     "evaluated_no_synthetic_gain",f"Six-hour storm-mean MAE (m): real Ridge {d(rr)}, hybrid {d(hh)}, persistence {d(ps)}.",
     "Hybrid lost to real Ridge at both leads and both flag scopes; unflagged persistence led at six hours. Uncertified issue-time clocks.",
     link(o+"verified/contrasts.csv"),link(o+"unflagged/contrasts.csv"))
nc,nurl,nref=later["adcirc_coordinates.json"]; ec,eurl,_=later["adcirc_land_screen.json"]
assert len(ec)==len(nc["stations"])==2
dist={k:v["nearest_8"][0]["distance_m"] for k,v in nc["stations"].items()}
assert dist=={"8723970":1639.25,"8724580":4028.78}
assert all(v["screen_radius_counts"]["1000"]==0 for v in nc["stations"].values())
assert all(len(v)==2 and all(x["land_features_intersecting_middle_90pct"]>0 for x in v.values()) for v in ec.values())
gate("adcirc_irma_collocation","ADCIRC+SWAN extraction and gauge matching","One storm (Irma), two Florida Keys stations",
     "blocked_no_physical_model_scored","Nearest points 1.639 and 4.029 km; none within 1 km; both straight segments intersect land in two NOAA land layers.",
     "Missing native depth, mesh connectivity, wet-state and datum tie. Feasibility gate only; not a physical skill result.",nurl,eurl)
for k,v in dist.items(): metric("adcirc_irma_collocation","nearest_distance",v,"m","NOAA station "+k,nurl)
metric("adcirc_irma_collocation","stations_with_node_within_1km",0,"stations","Two prespecified stations",nurl)

assert len({m["gate_id"] for m in gates})==len(gates)
assert {m["gate_id"] for m in gates}=={m["gate_id"] for m in metrics}
write("T7-synthetic-catastrophic-gates.csv",gates)
write("T7-gate-metrics.csv",metrics)
for m in models:
    path=m["method_url"].removeprefix(f"{G}/blob/{P}/")
    if path!=m["method_url"]: get(path)
write("SOURCE_MANIFEST.csv",list(inputs.values()))

def summary(title, headers, rows):
    esc=lambda a:str(a).replace("|","\\|")
    return "\n".join([title,"","| " + " | ".join(headers) + " |","| " + " | ".join(["---"]*len(headers)) + " |",
                      *["| " + " | ".join(map(esc,a)) + " |" for a in rows],""])

simple_names=["Frozen six-band router","Original frozen amount stack","Accepted low-band overlays",
 "Reported research-only mid-band overlay","Retrospective forward reference","Verified-link forward challenger",
 "Target-blind metadata candidate","Fixed 20× signed-amount candidate","Forward residual intervals","Fixed-split CQR",
 "Legacy RL budget simulation","Simulated transport optimization"]
selected=[0,1,2,3,6,7,10,11]
m=summary("**Table T3. Selected model lineage and evidence status.**",
          ["Version and status","Validation scope","Archived finding"],
          [[f"[{simple_names[i]}]({models[i]['evidence_url']})",models[i]["validation_method"],models[i]["manuscript_result"]]
           for i in selected])
m+=("\n*Table note.* Frozen Track A routed 830/912; four late rules used true obligations to gate held rows. "
    "All three historical amount treatments share that router. None of the reported scores alone validates an operational procedure. "
    "The matched forward and interval versions are also listed in the machine-readable extract and Tables T5 and T8.\n\n")
g=summary("**Table T7. Synthetic-transfer and catastrophic-label gates.**",
          ["Study","Disposition","Evidence"],[
          [f"[{a['study']}]({a['evidence_url']})",a["disposition"].replace("_"," "),a["manuscript_result"]] for a in gates])
g+=("\n*Table note.* A blocked fit supplies feasibility evidence, not an accuracy score. "
    "Reported physical endpoints and simulated damage do not supply historic FEMA funding labels. "
    "Previously examined signed obligations cannot establish independent gains; the fixed 20× candidate is archived, not accepted.\n")
(arg.out/"T3_T7_MANUSCRIPT_BLOCKS.md").write_text(m+g,encoding="utf-8")
print(f"PASS: {len(models)} T3 rows, {len(gates)} T7 dispositions, {len(metrics)} T7 metrics, {len(inputs)} pinned inputs")
