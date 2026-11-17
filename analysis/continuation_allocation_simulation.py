#!/usr/bin/env python3
"""Explicit synthetic allocation experiments, independent of FEMA funding.

No real disaster/need labels or operational benefits are implied. Fixed policies
and a fixed generator; common random numbers; chronological cash constraints.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/allocation';OUT.mkdir(parents=True,exist_ok=True)
SEED=20261117
MEAN_NEED=float(np.exp(2+.7**2/2))

def capped_proportional(cap,weights,budget):
    allocation=np.zeros(len(cap));remaining=max(float(budget),0.)
    for _ in range(len(cap)+1):
        active=allocation<cap-1e-10
        if not active.any() or remaining<=1e-10:break
        add=np.minimum(cap[active]-allocation[active],remaining*weights[active]/weights[active].sum())
        allocation[active]+=add;remaining-=add.sum()
    return allocation

def optimized(cap,weights,cash,fairness=0.):
    n=len(cap);floor=fairness*min(1.,cash[-1]/cap.sum())*cap
    # Scale floors until the early-prefix constraints are feasible as well.
    prefix=np.cumsum(floor);ratio=np.min(np.divide(cash,prefix,out=np.ones(n),where=prefix>0))
    floor*=min(1.,float(ratio))
    fit=linprog(-weights,A_ub=np.tril(np.ones((n,n))),b_ub=cash,
                bounds=list(zip(floor,cap)),method='highs')
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x

def sequential(estimate,cash,policy):
    n=len(estimate);alloc=np.zeros(n)
    for i in range(n):
        available=max(0.,float(cash[i]-alloc[:i].sum()))
        future=n-i-1
        if policy=='greedy_current':amount=available
        elif policy=='equal_pacing':amount=available/(future+1)
        elif policy=='forecast_reserve':amount=available*estimate[i]/(estimate[i]+future*MEAN_NEED)
        else:raise ValueError(policy)
        alloc[i]=min(float(estimate[i]),amount)
    return alloc

def metrics(a,need,vulnerability,cash,oracle):
    assert (a>=-1e-8).all() and np.all(np.cumsum(a)<=cash+1e-7)
    delivered=np.minimum(a,need);service=delivered/need
    value=float(np.dot(1+vulnerability,delivered))
    assert value<=oracle+1e-6
    high=vulnerability>=.6
    gap=float(service[high].mean()-service[~high].mean()) if high.any() and (~high).any() else np.nan
    return dict(unmet_need_units=float((need-delivered).sum()),wasted_units=float((a-delivered).sum()),
                fulfilled_fraction=float(delivered.sum()/need.sum()),weighted_delivered=value,
                min_disaster_service=float(service.min()),vulnerable_service_gap=gap,
                budget_utilization=float(a.sum()/cash[-1]),oracle_regret_units=max(oracle-value,0.))

def main():
    rng=np.random.default_rng(SEED);scenarios=[];results=[];allocations=[]
    # Predeclared 300 independent synthetic episodes, eight arrivals, three
    # planned budget ratios. Same episode inputs for all policy comparisons.
    for episode in range(300):
        n=8;estimate=rng.lognormal(2,.7,n);v=rng.uniform(.1,.95,n)
        sigma=.3+.5*v
        need=estimate*rng.lognormal(-.5*sigma**2,sigma,n)
        shock=float(rng.choice([.5,1.5]))
        for i in range(n):
            scenarios.append(dict(episode=episode,arrival=i,estimated_need_units=estimate[i],
                realized_need_units=need[i],synthetic_vulnerability=v[i],need_sigma=sigma[i],
                release_shock=shock,label='SYNTHETIC SCENARIO; no real disaster'))
        for ratio in (.4,.7,1.):
            planned=ratio*estimate.sum()
            cash=np.r_[np.repeat(.6*planned,4),np.repeat((.6+.4*shock)*planned,4)]
            upper=estimate*np.exp(norm.ppf(.9)*sigma-.5*sigma**2)
            for setting in ('simultaneous','sequential'):
                available=np.repeat(cash[-1],n) if setting=='simultaneous' else cash
                oracle_a=optimized(need,1+v,available)
                oracle=float(np.dot(1+v,oracle_a))
                if setting=='simultaneous':
                    policies={
                        'equal_capped':capped_proportional(estimate,np.ones(n),available[-1]),
                        'proportional_estimated_need':capped_proportional(estimate,estimate,available[-1]),
                        'robust_upper_need':capped_proportional(estimate,upper,available[-1]),
                        'vulnerability_LP':optimized(estimate,1+v,available),
                        'fairness_floor_LP':optimized(estimate,1+v,available,fairness=.5)}
                else:
                    policies={name:sequential(estimate,cash,name) for name in
                              ('greedy_current','equal_pacing','forecast_reserve')}
                for name,a in policies.items():
                    results.append(dict(episode=episode,budget_ratio=ratio,setting=setting,policy=name,
                                        **metrics(a,need,v,available,oracle)))
                    for i in range(n):
                        allocations.append(dict(episode=episode,budget_ratio=ratio,setting=setting,
                            policy=name,arrival=i,allocation_units=float(a[i]),
                            cash_available_cumulative=float(available[i]),realized_need_units=float(need[i])))
    frame=pd.DataFrame(results)
    frame.to_csv(OUT/'simulation_episode_metrics.csv',index=False)
    pd.DataFrame(scenarios).to_csv(OUT/'synthetic_scenarios.csv',index=False)
    pd.DataFrame(allocations).to_csv(OUT/'simulation_allocations.csv',index=False)
    metrics_cols=[c for c in frame.columns if c not in ('episode','budget_ratio','setting','policy')]
    summary=frame.groupby(['setting','budget_ratio','policy'])[metrics_cols].mean().reset_index()
    summary.to_csv(OUT/'simulation_policy_summary.csv',index=False)
    # Paired Monte Carlo sampling error only, never real-world confidence.
    deltas=[]
    for (setting,ratio),group in frame.groupby(['setting','budget_ratio']):
        baseline='proportional_estimated_need' if setting=='simultaneous' else 'equal_pacing'
        pivot=group.pivot(index='episode',columns='policy',values='unmet_need_units')
        for name in pivot:
            diff=(pivot[name]-pivot[baseline]).to_numpy()
            se=diff.std(ddof=1)/np.sqrt(len(diff))
            deltas.append(dict(setting=setting,budget_ratio=ratio,policy=name,baseline=baseline,
                paired_mean_unmet_delta=float(diff.mean()),mc95_lower=float(diff.mean()-1.96*se),
                mc95_upper=float(diff.mean()+1.96*se),interpretation='Monte Carlo error under specified synthetic generator only'))
    pd.DataFrame(deltas).to_csv(OUT/'paired_monte_carlo_differences.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(seed=SEED,episodes=300,arrivals_per_episode=8,
        evaluations=len(frame),allocation_rows=len(allocations),all_cash_constraints_verified=True,
        data_classification='SYNTHETIC SCENARIO',historical_funding_used=False,
        need_model='estimated lognormal(2,.7); realized independent mean-preserving lognormal multiplicative error sigma=.3+.5*vulnerability',
        budgets='planned ratios .4/.7/1 of estimated total need; .6 initially, .4 times .5/1.5 shock at arrival4',
        fairness='one operational illustration: floors half of equal estimated service fraction; policy choice requires human governance',
        oracle='hindsight need-aware LP; sequential prefix cash constraints; descriptive upper bound, not deployable policy',
        limitations='one synthetic generator; policies are illustrative; no causal benefit, empirical need validity, or optimal human welfare claim'),indent=2))
    print('SIMULATION_COMPLETE',len(frame),'policy evaluations; all cash and regret invariants passed')

if __name__=='__main__':main()
