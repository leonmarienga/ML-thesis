"""Fold-local oversampling for retrospective metadata. No synthetic FEMA records."""
from hashlib import sha256
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from thesis_forward_models import CATS, META
from thesis_hazard_features import CUTS

CATEGORY = [*CATS, "incident_month", *META[2:]]
DELAY = "declaration_delay_days"
PREDICTORS = [*CATEGORY, DELAY]
TAIL_BANDS = (3, 4, 5)
RATIOS = (2, 5, 10, 20)
METHODS = ("weight", "random", "smote", "smotenc", "smoter", "smogn", "copula", "hybrid")
CLASS_ONLY = {"smote", "smotenc"}
CONTINUOUS_BANDS = set(TAIL_BANDS)
SEED = 91831


def key_seed(key):
    return int.from_bytes(sha256(f"{SEED}|{key}".encode()).digest()[:4], "little")


def report_support(train):
    tail = train[train.actual_band.isin(TAIL_BANDS)]
    cells = tail.groupby(["actual_band", "incidentType", "state"]).event_group.nunique()
    exact = tail.groupby(["actual_band", *CATEGORY], dropna=False).event_group.nunique()
    rows = []
    for band in TAIL_BANDS:
        band_rows = tail[tail.actual_band.eq(band)]
        groups = cells[cells.index.get_level_values("actual_band") == band]
        strict = exact[exact.index.get_level_values("actual_band") == band]
        rows.append({"band": band, "real_rows": len(band_rows),
            "unique_episodes": band_rows.event_group.nunique(),
            "max_episodes_per_state_hazard": int(groups.max()) if len(groups) else 0,
            "state_hazard_strata_with_two": int(groups.ge(2).sum()),
            "strict_cells_with_two": int(strict.ge(2).sum()),
            "state_hazard_strata_copula_gate": int(groups.ge(8).sum()),
            "state_hazard_strata_ctgan_gate": int(groups.ge(100).sum())})
    return rows


def metadata_tuple(row):
    return tuple(row[name] for name in CATEGORY)


def distance(a, b, delay_scale, mixed=True):
    continuous = abs(a[DELAY]-b[DELAY])/delay_scale
    mismatch = np.mean([a[cat]!=b[cat] for cat in CATEGORY[3:]]) if mixed else 0.0
    return float(np.hypot(continuous, mismatch))


def gaussian_empirical(pool, allocator, rng):
    """Conditional empirical marginals with regularized normal-score dependence."""
    values = pool[[DELAY,"target"]].to_numpy(float, copy=True)
    values[:,1] = np.log1p(values[:,1])
    if np.any(np.ptp(values,axis=0) == 0):
        return []
    z=np.column_stack([norm.ppf(rankdata(column)/(len(column)+1)) for column in values.T])
    cov=.7*np.corrcoef(z.T)+.3*np.eye(2)
    u=norm.cdf(rng.multivariate_normal(np.zeros(2),cov,size=allocator))
    return [(float(np.quantile(values[:,0],x)),float(np.expm1(np.quantile(values[:,1],y)))) for x,y in u]


def add_rows(train, method, ratio, identity, seed_namespace):
    """Return generated feature rows, source lineage and per-stratum diagnostics."""
    trial = f"{identity}|{method}|{ratio}"
    seed = key_seed(f"{seed_namespace}|{method}|{ratio}")
    real = train[train.actual_band.isin(TAIL_BANDS)].copy()
    source = real.to_dict("records")
    reference = {(metadata_tuple(r),round(r[DELAY],9),int(r["actual_band"])) for r in source}
    reference_joint = {(metadata_tuple(r),round(r[DELAY],9),round(r["target"],5)) for r in source}
    delay_scale = max(float(train[DELAY].quantile(.75)-train[DELAY].quantile(.25)),1.)
    audit, generated = [], []
    if method in ("real", "weight"):
        return pd.DataFrame(), [{
            "trial_id":trial,"seed_namespace":seed_namespace,"seed":seed,"band":-1,"state":"","hazard":"","ratio":ratio,
            "real_rows":len(real), "real_episodes":real.event_group.nunique(),
            "used_ratio":ratio, "attempts":0,"accepted":0,"status":"real_weights" if method=="weight" else "real",
            "source_ids":""
        }]
    by = ["actual_band"] if method=="random" else ["actual_band","incidentType","state"]
    for _, pool in real.groupby(by):
        rows = pool.to_dict("records")
        band = int(rows[0]["actual_band"])
        geo = rows[0]["state"] if method!="random" else "all"
        hazard = rows[0]["incidentType"] if method!="random" else "all"
        distinct=pool.event_group.nunique()
        # Use the same draw prefix across ratios. Requests capped at the same
        # effective size produce identical feature rows and amount proxies.
        seed=key_seed(f"{seed_namespace}|{method}|{band}|{geo}|{hazard}")
        rng=np.random.default_rng(seed)
        nominal = {metadata_tuple(r) for r in rows}
        count_before=len(generated)
        # Novel generation is deliberately kept small when there are only two episodes.
        used = min(ratio, 5 if distinct<4 else 10 if distinct<8 else 20) if method!="random" else ratio
        required = 8 if method=="copula" else 1 if method=="random" else 2
        requested = (used-1)*len(rows)
        donors = {}
        mixed = method!="smote"
        for i, row in enumerate(rows):
            candidate = [j for j, entry in enumerate(rows) if entry["event_group"] != row["event_group"]
                         and (mixed or metadata_tuple(entry)==metadata_tuple(row))]
            donors[i] = sorted(candidate,key=lambda j:(distance(row,rows[j],delay_scale,mixed),rows[j]["disasterNumber"]))[:5]
        fit_pairs=[i for i,j in donors.items() if j]
        available = distinct>=required and (bool(fit_pairs) or method in ("random","copula"))
        independent_distances = [distance(rows[i],rows[j],delay_scale) for i,j in combinations(range(len(rows)),2)
                                 if rows[i]["event_group"]!=rows[j]["event_group"]]
        near_threshold=.5*np.median(independent_distances) if independent_distances else 0.
        copula = gaussian_empirical(pool,requested,rng) if available and method=="copula" else []
        if method=="copula" and not copula:
            available=False
        attempted, duplicate_rejected = 0, 0
        del_min,del_max = float(pool[DELAY].min()),float(pool[DELAY].max())
        y_min,y_max = float(pool.target.min()),float(pool.target.max())
        if available:
            for number in range(requested):
                attempted+=1
                i=int(rng.choice(fit_pairs)) if method not in ("random","copula") else int(rng.integers(len(rows)))
                anchor=rows[i]
                neighbor=anchor
                nominal_parent=anchor
                delay=anchor[DELAY]
                y=anchor["target"]
                branch="repeat"
                if method!="random" and method!="copula":
                    j=int(rng.choice(donors[i]))
                    neighbor=rows[j]
                    # Select an observed *joint* nominal mode, not independently mixed flags.
                    peers=[rows[k] for k in [i,*donors[i]]]
                    votes={metadata_tuple(p):sum(metadata_tuple(p)==metadata_tuple(q) for q in peers) for p in peers}
                    choices=[p for p in peers if votes[metadata_tuple(p)]==max(votes.values())]
                    nominal_parent=choices[int(rng.integers(len(choices)))] if method=="smotenc" else (anchor if rng.random()<.5 else neighbor)
                    alpha=float(rng.uniform(.05,.95))
                    delay=(1-alpha)*anchor[DELAY]+alpha*neighbor[DELAY]
                    y=float(np.expm1((1-alpha)*np.log1p(anchor["target"])+alpha*np.log1p(neighbor["target"])))
                    branch="interpolation"
                    if method=="smogn" and distance(anchor,neighbor,delay_scale)>near_threshold:
                        delay=float(np.clip(rng.normal(anchor[DELAY],.05*pool[DELAY].std(ddof=0)),del_min,del_max))
                        original_log=np.log1p(anchor["target"])
                        sigma=.05*np.log1p(pool.target).std(ddof=0)
                        y=float(np.clip(np.expm1(rng.normal(original_log,sigma)),y_min,y_max))
                        nominal_parent=anchor
                        branch="gaussian_noise"
                    if method in CLASS_ONLY:
                        y=None
                elif method=="copula":
                    delay,y=copula[number]
                    branch="empirical_gaussian_copula"
                features={col:nominal_parent[col] for col in CATEGORY}
                features[DELAY]=float(delay)
                category=metadata_tuple(features)
                same_band= y is None or int(np.searchsorted(CUTS,y,side="right"))==band
                assert category in nominal and same_band and del_min-1e-10<=delay<=del_max+1e-10
                assert y is None or y_min-1e-5<=y<=y_max+1e-5
                reused=(category,round(delay,9),band) in reference
                joint=y is not None and (category,round(delay,9),round(y,5)) in reference_joint
                if method!="random" and ((method in CLASS_ONLY and reused) or (method not in CLASS_ONLY and joint)):
                    duplicate_rejected+=1
                    continue
                row={**features,"target":y,"actual_band":band,"synthetic_id":f"{sha256(trial.encode()).hexdigest()[:12]}-{len(generated)+1:04d}",
                     "method":method,"trial_id":trial,"generation":branch,"seed":seed,
                     "source_a":anchor["disasterNumber"],"source_b":neighbor["disasterNumber"],
                     "episode_a":anchor["event_group"],"episode_b":neighbor["event_group"],
                     "category_source":nominal_parent["disasterNumber"],"feature_duplicate":reused,"joint_duplicate":joint,
                     "min_distance_to_real_stratum":min(distance(features,r,delay_scale) for r in rows),
                     "fold":identity,"county_or_coordinates_synthesized":False,"uses_real_fema_identifier":False}
                generated.append(row)
        gained=len(generated)-count_before
        generated_in_cell=generated[count_before:]
        real_correlation=pool[[DELAY,"target"]].corr(method="spearman").iloc[0,1] if len(pool)>1 else np.nan
        synthetic_correlation=np.nan
        if len(generated_in_cell)>1 and method not in CLASS_ONLY:
            sample=pd.DataFrame(generated_in_cell)
            synthetic_correlation=sample[[DELAY,"target"]].corr(method="spearman").iloc[0,1]
        audit.append({"trial_id":trial,"seed_namespace":seed_namespace,"seed":seed,"band":band,"state":geo,"hazard":hazard,"ratio":ratio,
            "real_rows":len(rows),"real_episodes":distinct,"used_ratio":used,"attempts":attempted,"accepted":gained,
            "rejected_real_duplicates":duplicate_rejected,"status":"generated" if gained else "no_novel_rows" if available else "gate_closed",
            "source_ids":"|".join(map(str,sorted(pool.disasterNumber))),
            "source_episodes":"|".join(map(str,sorted(pool.event_group.unique()))),
            "training_delay_min":del_min,"training_delay_max":del_max,
            "real_delay_target_spearman":real_correlation,"synthetic_delay_target_spearman":synthetic_correlation,
            "training_target_min":y_min,"training_target_max":y_max})
    return pd.DataFrame(generated),audit
