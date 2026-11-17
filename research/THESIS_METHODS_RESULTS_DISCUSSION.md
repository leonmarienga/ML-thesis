# Thesis-ready draft: methodology, results and discussion

This draft reports completed and historical evidence with explicit distinctions.
It requires researcher review and completion of the gaps in FINAL_CONCLUSION.md
before being presented as a complete final thesis.

## Methodology

The research separates retrospective financial prediction from physical hazard
measurement and relief allocation. The original accounting dataset contains 971
FEMA disasters from fiscal2010–2024, of which912 are non-Biological. Historical
obligations form the prediction target. They do not directly measure losses,
humanitarian needs, affected population or required relief.

Track A was preserved as a frozen historical benchmark. Six target bands are
defined by lower-inclusive thresholds0,100,000,1,000,000,50,000,000,200,000,000 and
500,000,000 dollars, with negative net funding clamped to zero. Its original
leave-fiscal-year-out predictions were not retrained. The exact original per-row
archive was recovered through the GitHub connector, its ZIP SHA256 verified
against GitHub's digest, and its metrics recomputed. LFYO differs from forward-year evaluation
because subsequent years can enter a held-year training set.

Phase 50 examined whether prior water-change candidates could reflect seasonal
or interannual surface-water variation. County/site keys were fixed from the
historical execution log. Sixty-three original anchors and four additional
Minnesota windows were reconstructed using the committed deterministic county
geometry algorithm and Census2023 polygons. Site selection did not use funding,
new imagery or model errors. A discovered incident-date discrepancy was resolved
using official FEMA metadata and historical raster-log consistency.

Monthly JRC v1.5 classified pixels for2022–2024 were retrieved. Class0 was treated
as unobserved throughout; classes1 and 2 represented dry and water observations.
All monthly grids and polygon masks were required to align. Primary seasonal
comparisons contrasted preceding-month-to-incident-month changes in the event
year with the corresponding change in the prior year. The2024 interannual audit
contrasted2023-to-2024 with 2022-to-2023 at the same calendar month. Comparisons
crossing the source-edition boundary would be excluded, with an explicit test
for January2023. No such excluded comparison occurred in the final fixed cohort.

The primary estimand was the difference in classified water gains per100 km²
jointly observable area, using pixels observed at every required date. Loss and
net change were also recorded. Ellipsoidal pixel areas were calculated per raster
row. Thirty-percent common observability was the primary threshold;60% and 80%
and pair-specific denominators were fixed sensitivity analyses. Aggregates used
the same eligible sites in both arms. Minnesota expansion and removal were
reported separately. These temporal controls were not assumed flood-free.

An independent-sensor feasibility check retrieved NASA MODIS flood-product
renders for the six Minnesota windows on three prespecified dates spanning the
regional NWS flood chronology. These250m-product visualizations were evaluated
as source evidence, not as raw QA arrays or surveyed inundation labels. Broad
hazard availability was investigated using an official NOAA2024 storm archive,
an Atlantic NHC best-track archive, eight USGS earthquake catalog queries and
MTBS documentation/service checks. Candidate geographic and temporal links
were kept separate from independently verified event attribution.

Allocation experiments used synthetic need rather than historical obligations.
Three hundred independently generated episodes contained eight arrivals each.
Estimated need followed a lognormal distribution; realized need used a
mean-preserving multiplicative error with uncertainty increasing with synthetic
vulnerability. Planned budget ratios0.4,0.7 and 1.0 were tested with uncertain
mid-episode supplementary releases. Five simultaneous and three sequential
policies were fixed before examining outcomes. Linear programs optimized
weighted delivered units under allocation and cumulative cash constraints.
Fairness floors represented one explicit normative scenario. Hindsight regret
was measured against an oracle obeying the same sequential cash limits.
The policy named robust_upper_need is an upper-quantile-weighted proportional
heuristic, not a formal worst-case robust-optimization guarantee.

## Results

The original data include 430 severe storms,198 floods,108 hurricanes,33 fires,
27 tornadoes and 8 earthquakes; no Drought records exist. The recovered Track A
log reports830/912 routing successes, MAE$5.014819M and R²0.88677657. These remain
retrospective historical results, not new operational validation.

Phase 50 successfully read all 289 planned monthly windows. At30% common
observability,44 primary anchor windows covered 27 of 36 disasters. The seasonal
event and placebo gain rates were0.080857 and 0.027784 km² per100 observed km²,
respectively, giving an excess 0.053073. Removing Minnesota yielded0.054403.
Adding four Minnesota sites raised the excess to 0.687342. The2024 interannual
audit covered 17 sites and 11 disasters, with excess gain 0.086369. Coverage and
normalization therefore materially affected the usable evaluation population.

DR 4816’s workbook start date differed from official FEMA metadata by two months.
Correcting only the new observational audit restored historical log agreement.
All65 comparable site reads reproduced the logged observability and rounded
gain; two Alaska sites initially lacked a usable historical log comparator.
Subsequent recovery of all original geospatial archives verified every one of
the 187 geometries exactly and all 67 Phase50 integer gain counts and event
observation fractions, including Alaska. Only Alaska rows were replaced when
merging the correction: 561 monthly records and 374 comparisons remain unique.

All18 requested NASA renders were obtained. The map products show flood-class
colors at some river-edge Minnesota extra windows and insufficient data across
many inland views. Because these are coarse classifications without raw QA or
survey truth, no flood-label performance metric or validated exposure was derived.
The source checks established physical-field availability in NOAA wind/hail/
tornado records and actual USGS/NHC data retrieval, while leaving disaster
matching uncertified. Numeric snowfall/ice intensity was absent from the
inspected storm archive fields. The MTBS map service returned503 despite
successful access to its authoritative documentation.

The allocation experiment completed7,200 policy evaluations. All cash-feasibility
and oracle-bound assertions passed. At planned budget ratio 0.7, sequential greedy
allocation achieved mean fulfilled fraction0.5785 but minimum-disaster service
0.0604, versus 0.5583 and 0.2717 for forecast reserve. Under simultaneous allocation,
proportional estimated need achieved fulfilled fraction0.6449; vulnerability
priority LP achieved0.5585 and a larger service gap favoring the more vulnerable
synthetic group. None of these results estimates real unmet humanitarian need.

## Discussion and interpretation

The evidence supports preserving the historical predictor while rejecting
premature integration of local classified water changes as total disaster
severity. Strong spatial sensitivity and cloud/snow-related observability changes
limit extrapolation. Agreement with an independent coarse satellite product
can corroborate contextual plausibility without supplying independent truth.
The metadata discrepancy demonstrates why source linkage and date auditing are
part of measurement validity rather than merely engineering cleanup.

Prior flood-model enrichment results remain exploratory: the historically
reported NOAA reporting-density model improved over a FEMA-only comparator on
59 repeatedly examined tests, but subsequent physical rainfall/flow additions
did not beat it. Reporting density is not a direct physical severity measure.
No new funding-model improvement is established by this continuation. Further
model complexity cannot resolve uncertified labels, retrospective feature
availability or the absence of a genuinely untouched outcome population.

Previously reported uncertainty experiments illustrate that high pooled coverage
does not guarantee catastrophic reliability. Very small rare-event samples and
large interval widths preclude reliable conditional guarantees. Track B quantile
models cannot be represented as uncertainty automatically attached to Track A.

The simulation shows a decision problem separate from accounting prediction.
Efficiency, cash utilization and protection of the worst-served disaster can
conflict. Vulnerability weighting expresses a policy preference; optimizing that
objective does not prove a universally fair allocation. Real adoption requires
independently justified need labels, domain-specific utilities, verified inputs,
publicly accountable priorities and prospective evaluation.

## Contributions, novelty and limitations

The demonstrated contributions are an auditable continuation from an interrupted
experiment, fixed-window temporal placebo methodology with common-pixel masks,
an identified date error, explicit spatial-influence analysis, source/feature
provenance, offline reproduction and a need-separated allocation simulation.
These are project-specific empirical and reproducibility contributions. Surface
water mapping and conformalized quantile regression are established methods;
their original publication metadata were verified. A bounded search also found
prior disaster decision-support and stochastic-allocation research, but a
comprehensive full-text novelty review is incomplete. No first-of-kind claim
is warranted.

The study lacks separately verified trained-model weights, validated broad
flood/other-hazard footprints, contemporaneous
feature snapshots, independent future outcomes and real need labels. Source
checks were bounded pilots rather than a complete multi-hazard data assembly.
The requested final operational framework therefore remains unvalidated and
the full thesis research objective remains incomplete. The next decisive work
is raw historical product/QA ingestion, certified event
linkage and prospectively frozen evaluation, not further tuning of examined cases.
