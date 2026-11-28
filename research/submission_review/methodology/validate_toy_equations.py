"""Numerical checks linked to source functions, executed only on synthetic fixtures."""
import argparse
import ast
import dataclasses
from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

SHA = "b0b6edb229d112dd6d11896a512bf1112414e4f6"
REPORT = {}


def evaluate(source, names, **deps):
    # No imports, __main__ blocks or IO from the notebook/script are executed.
    tree = ast.parse(source)
    selected, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            hits = {item.id for item in node.targets[0].elts if isinstance(item, ast.Name)} & set(names)
            if not hits:
                continue
            assert len(hits) == 1  # tuple assignment in pinned transport constants
            name = hits.pop()
        else:
            continue
        if name in names:
            found.add(name)
            selected.append(node)
    assert found == set(names), set(names) - found
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<pinned source excerpt>", "exec"), deps)
    return deps


def check_labels(read):
    # Evaluate original clipping/indexing lines as an isolated block.
    a = read("analysis/thesis_hazard_features.py").splitlines()
    m = evaluate("\n".join(a), ["CUTS"])
    cuts = np.asarray(m["CUTS"])
    w = [-1., 0., *[val for cut in cuts for val in (cut-0.01, cut)]]
    m.update(np=np,d=pd.DataFrame({"totalObligatedFunding": w}))
    exec("\n".join(line.strip() for line in a[54:56]), m)
    assert m["d"].actual_band.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert list(m["d"].target.head(2)) == [0., 0.]
    REPORT["E1"] = {"exact_boundaries_and_negative_clipping": "passed", "bands": m["d"].actual_band.tolist()}

    names = ["aggregate"]
    src = evaluate(read("analysis/thesis_historical_coverage.py"), names, np=np, pd=pd)
    actions = pd.DataFrame([
        (1, "m1", 0, 901, 10., "2019-01-30", "2019-01-30"),
        (2, "m1", 1, 901, 30., "2019-01-31", "2019-01-31"),
        (3, "m1", 2, 901, -45., "2022-01-31", "2022-03-02"),
        (4, "m1", 3, 901, 99., "2022-02-01", "2022-02-01"),
        (5, "m2", 0, 902, 4., None, "2019-02-01"),
        (6, "m3", 0, 903, 8., "2020-02-01", "2020-02-01"),
        (6, "m3", 0, 903, 8., "2020-02-01", "2020-02-01"),
    ], columns="actionId maId maAmendNumber disasterNumber obligationAmount dateObligated dateReceived".split())
    dates = pd.DataFrame({"disasterNumber": [901, 902, 903, 904],
                          "declaration_date": pd.to_datetime(["2019-01-31"] * 4, utc=True)})
    signed = src["aggregate"](actions, dates).set_index("disasterNumber")
    assert signed.loc[901,"net_through_36m_usd"] == -5.
    assert signed.loc[901,"net_obligation_usd"] == 94.
    assert signed.snapshot_status.tolist() == ["published_actions","missing_obligation_date",
                                               "ambiguous_action_structure","no_public_rows"]
    maturity = dates.declaration_date.iloc[0]+pd.DateOffset(months=36)+timedelta(days=90)
    dat = read("analysis/thesis_obligation_horizon_audit.py")
    # Pin equality operators in these two separate gate sites; test the extracted expressions.
    assert 'prior.release_review_date.lt(first)' in dat and 'rollup.release_review_date.le(snapshot)' in dat
    x = signed.reset_index().assign(release_review_date=[maturity]*4)
    v = {"prior": x, "rollup":x, "first":maturity, "snapshot":maturity}
    assert not eval("prior.release_review_date.lt(first) & prior.snapshot_status.eq('published_actions')",v).any()
    assert eval("rollup.snapshot_status.eq('published_actions') & rollup.release_review_date.le(snapshot)",v).sum()==1
    REPORT["E2"] = {"signed_36m": -5, "all_published_action_sum":94, "maturity_equality_snapshot_not_origin": True}


def check_forward(read):
    source = read("analysis/thesis_forward_models.py")
    env = evaluate(source, ["split","paired_differences","SEED"], np=np, pd=pd)
    d = pd.DataFrame([
        *[(i,2018,"2016-01-01","2016-02-01","2016-01-03",i) for i in range(1,42)],
        (42,2018,"2016-01-01",None,"2016-01-03",42),  # end falls back to start
        (43,2018,"2016-01-01","2019-01-01","2016-01-03",43),  # exactly held cutoff
        (44,2018,"2016-01-01","2016-02-01","2016-01-03",1001), # shared episode
        (45,2020,"2016-01-01","2016-02-01","2016-01-03",45),  # future FY
        (1001,2019,"2019-01-02","2019-01-08","2019-01-01",1001), # declaration determines cutoff
        (1002,2019,"2019-01-04","2019-01-08","2019-01-05",1002),
    ], columns=["disasterNumber","fyDeclared","start","end","declaration","event_group"])
    for k in ("start","end","declaration"):
        d[k] = pd.to_datetime(d[k])
    tr, held = env["split"](d,2019)
    assert tr.disasterNumber.tolist() == list(range(1,43))
    assert held.disasterNumber.tolist() == [1001,1002]
    REPORT["E3"] = {"train_n": len(tr), "held_n": len(held),
                    "strict_date_shared_episode_and_future_fiscal_year_exclusion": "passed"}

    line = next(l.strip() for l in source.splitlines() if "amount = np.expm1" in l)
    env.update(regression=SimpleNamespace(predict=lambda b:np.log1p(np.array([-.5, 18.,200.]))),
               b=None, train=pd.DataFrame({"target":[0,20]}))
    exec(line,env)
    np.testing.assert_allclose(env["amount"], [0.,18.,20.])
    b = pd.DataFrame({"disasterNumber":[1,2,3],"event_group":[1,1,2],
                      "target":[1.,1.,10.],"actual_band":[0,0,1],
                      "predicted_amount":[2.,3.,10.],"predicted_band":[0,1,1],
                      "variant":"ablation_retrospective_metadata"})
    a = b.assign(predicted_amount=[2.,6.,13.],predicted_band=[0,0,0], variant="toy")
    with tempfile.TemporaryDirectory() as p:
        env["paired_differences"](pd.concat([a,b],ignore_index=True),Path(p))
        boot = pd.read_csv(Path(p)/"paired_event_bootstrap.csv").set_index("variant").loc["toy"]
    # Error differences [0,3] for group 1, [3] for group 2.
    # Possible replacement samples have declaration weights 4, 3 or 2.
    assert boot.mae_difference_usd == 2.
    assert boot.mae_ci_low == 1.5 and boot.mae_ci_high == 3. and boot.clusters==2 and boot.draws==3000
    REPORT["E4"] = {"capped_back_transform":[0.,18.,20.],"paired_mae_delta":2.,"toy_interval":[1.5,3.]}


def check_intervals(read):
    import math
    a = evaluate(read("analysis/thesis_uncertainty.py"),["conformal_quantile","intervals","VARIANTS","LEVELS"],
                 ceil=math.ceil,np=np,pd=pd)
    b = evaluate(read("analysis/cqr_track_b_fixed_split.py"),["qstat"],math=math,np=np)
    for q in [a["conformal_quantile"],b["qstat"]]:
        assert q(np.array([1.,2.,3.,4.]),.8)==4.
        assert np.isinf(q(np.array([1.,2.,3.,4.]),.9))
        assert q(np.array([-5.,-4.,-3.,-2.]),.8)==-2.
    d = pd.DataFrame({
        "variant":["nested_metadata"]*35,
        "disasterNumber":np.arange(35),
        "fyDeclared":[2022]*32 + [2023]*3,
        "event_group":np.r_[np.arange(32),0,32,33],
        "incidentType":["Synthetic"]*35,
        "actual_band":[0]*35, "predicted_band":[0]*35,
        "target":[1000.] + [2.]*31 + [4.,0.,6.],
        "predicted_amount":[0.]*32 + [2.]*3, "top_band_probability":[.9]*35,
    })
    results=a["intervals"](d)
    one=results[(results.method=="dollar") & (results.level==.8)]
    assert one.calibration_n.eq(31).all() and one.score_quantile.eq(2.).all()
    assert one.lower_usd.eq(0.).all() and one.upper_usd.eq(4.).all()
    assert one.covered.tolist() == [True, True,False]
    REPORT["E5"] = {"conformal_rank_finite_and_infinite": "passed", "CQR_negative_score": -2.,
                    "residual_same_group_excluded_calibration_n":31, "toy_inclusive_coverage":2/3}


def check_logistics(read):
    service_code=read("analysis/thesis_service_scenarios.py")
    solver=evaluate(read("analysis/thesis_water_recourse.py"),["optimize"],linprog=linprog,np=np)["optimize"]
    s = evaluate(service_code,["HOUSEHOLDS","WEIGHT","VULNERABILITY","HIGH","DRAWS","proportional","dispatch",
                               "recourse","verify","summarized"],np=np,optimize=solver)
    first=np.ones(8)*4
    later=np.ones(8)*5
    branch=np.tile(later,(s["DRAWS"],1))
    rep=np.full(s["DRAWS"],12.)
    stock=12.
    xs={}
    for risk in (False,True):
        tx=s["recourse"](first,stock,branch,rep,risk,floor=.2)
        left=max(0.,stock-tx.sum())+12.
        ty=s["dispatch"](later,left,False)
        s["verify"](first,later,tx,ty,stock,left)
        assert np.all(tx+1e-8>=.2*s["proportional"](first,stock))
        assert abs(tx.sum()+ty.sum()-24.)<1e-6
        xs[str(risk)]=float(tx.sum())
    f=pd.DataFrame({"zone":np.arange(8),"need":[10]*8,"served":[0,1,2,3,4,5,6,7]})
    coverage=s["summarized"](f)
    assert abs(coverage["served_fraction"]-.35)<1e-12
    assert abs(coverage["high_vulnerability_served_fraction"]-.60)<1e-12
    assert coverage["lowest_zone_fraction"]==0.
    REPORT["E6"]={"stock_budget_and_proportional_floor":"passed","first_wave_allocations_by_risk":xs,
                  "simulated_overall_high_and_minimum_coverage":coverage}

    t = evaluate(read("analysis/thesis_transport_pareto.py"),
        ["CARGO","VOLUME","SERVICES","SIZE","FUTURES","STOCK_MATRIX","ROUTE_MATRIX","VOLUME_ROW",
         "VULNERABILITY_WEIGHT","Stage","proportional","limits","constraints","bounds","solve","dispatch","recourse","verify"],
        np=np,linprog=linprog,dataclass=dataclasses.dataclass,ZONES=8,VULNERABILITY=s["VULNERABILITY"])
    need=np.full((5,8),4.)
    inlet=np.full(5,20.)
    routes=np.r_[0.,np.full(7,10.)]
    p0=t["Stage"](need,inlet,routes,np.full((5,8),.5),90.)
    p1=t["Stage"](2*need,inlet,2*routes,np.full((5,8),.8),190.)
    flows={}
    for name in ("efficiency","vulnerability","geography","mixed","maxmin","expected","cvar","worst_branch"):
        x=t["recourse"](p0,[p1]*4,name,.25) if name in {"expected","cvar","worst_branch"} else t["dispatch"](p0,name,.25)
        a=t["verify"](p0,x,.25)
        assert abs(a[:,0].sum())<1e-9 and need[:,0].sum()>0
        np.testing.assert_allclose(a,.5*x)
        assert np.min(a.sum(0)/need.sum(0))==0. # closed but counts in achieved fairness
        flows[name]=float(x.sum())
    REPORT["E7_transport"]={"all_8_objectives_toy_feasibility":"passed","closed_route_served":0.,
                            "closed_route_requested":20.,"first_wave_shipment_totals":flows}


def check_rl(read):
    n=json.loads(read("reinforcement_learning.ipynb"))
    read_cell=lambda i:"".join(n["cells"][i]["source"])
    class Env:
        def reset(self,seed=None): pass
    def box(**k):
        return SimpleNamespace(**k)
    constants={"ACTION_MULTIPLIERS","TOTAL_BUDGET","INITIAL_RELEASED_BUDGET","FUNDING_RELEASE_SCHEDULE","SCENARIOS_PER_PAGE"}
    e=evaluate(read_cell(6),constants)
    e=evaluate(read_cell(9),["calculate_reward"],np=np,**{k:e[k] for k in constants})
    e=evaluate(read_cell(11),["CleanBudgetAdjustmentEnv"],np=np,pd=pd,gym=SimpleNamespace(Env=Env),
               spaces=SimpleNamespace(Box=box,Discrete=lambda n:n),**{k:e[k] for k in constants|{"calculate_reward"}})
    g=e["CleanBudgetAdjustmentEnv"](io.StringIO("ml_base_prediction,true_funding\n100,900\n100,900\n"),
              total_budget=1000,initial_released_budget=10,funding_release_schedule={1:10,2:20},
              scenarios_per_page=1,episode_length=2)
    obs,_=g.reset(seed=1)
    np.testing.assert_allclose(obs,[np.log1p(100)/25,1.,10/8e9,0.,10.,10.],rtol=1e-6)
    obs1,r1,done,_,info=g.step(5)
    assert not done and info["final_recommendation"]==10. and info["released_budget_so_far"]==10.
    np.testing.assert_allclose(obs1,[np.log1p(100)/25,0.,10/8e9,1.,10.,10.],rtol=1e-6)
    obs2,r2,done,_,info=g.step(0)
    assert done and info["final_recommendation"]==20. and info["released_budget_so_far"]==30.
    np.testing.assert_allclose(obs2,0.)
    np.testing.assert_allclose(r1,-abs(np.log1p(10)-np.log1p(900))-1.5-.5)
    np.testing.assert_allclose(r2,-abs(np.log1p(20)-np.log1p(900))-1.5-.5)
    fn=e["calculate_reward"]
    np.testing.assert_allclose(fn(2.,4.,15.,100.,20.,1,2,True),-abs(np.log1p(2.)-np.log1p(4.))-.25*15./20.)
    assert e["ACTION_MULTIPLIERS"] == [.40,.55,.70,.85,1.0,1.10]
    REPORT["E7_RL"]={"stale_observation_at_release_boundary":True,"post_release_overdraft_impossible":True,
                     "toy_steps_usd":[10.,20.],"terminal_unused_budget_penalty_checked":True,
                     "toy_rewards":[float(r1),float(r2)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("report",type=Path)
    a=p.parse_args()
    found=subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"],text=True).strip()
    assert found==SHA
    # git show reads pinned tracked source rather than a possibly edited working-tree file.
    read=lambda name:subprocess.check_output(["git","-C",str(a.repo),"show",f"{SHA}:{name}"],text=True)
    funcs=[check_labels,check_forward,check_intervals,check_logistics,check_rl]
    for f in funcs:
        f(read)
    d={"pin":found,"synthetic_fixtures_only":True,"historical_outcomes_loaded":False,"model_training":False,
       "numerical_checks":REPORT}
    a.report.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    print(json.dumps(d,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
