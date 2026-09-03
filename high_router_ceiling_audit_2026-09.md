# High-value router ceiling audit

## Objective

Operational high-value bands:

- Band 3: $50M-$200M, n=39, required recall >=32/39 = 82.05%.
- Band 4: $200M-$500M, n=9, required recall >=8/9 = 88.89%.
- Band 5: $500M+, n=7, required recall >=6/7 = 85.71%.

The thesis acceptance requirement is >=80% recall for every final operational band. Because of the small rare-band counts, the exact integer requirements above are used.

## What has passed individually

### Band 3 vs Band 4 boundary

A nested leave-fiscal-year-out exposure ExtraTrees gate achieved:

- Band 3 recall: 34/39 = 87.18% in one nested screen; a separately reproduced configuration achieved 33/39 = 84.62%.
- Band 4 recall: 8/9 = 88.89%.

This demonstrates that the $50M-$200M / $200M-$500M boundary is separable with target-free exposure, mission-scale, geographic-scale, and event-relative features.

### Band 4 vs Band 5 boundary

A nested leave-fiscal-year-out 3-nearest-neighbor gate based on log mission count and Normal-priority mission share, blended with declaration-year context selected inside the outer training fold, achieved:

- Band 4 recall: 8/9 = 88.89%.
- Band 5 recall: 6/7 = 85.71%.

This demonstrates that the $200M-$500M / $500M+ boundary can also clear the integer 80% requirement when evaluated locally.

## Why the final 3-band high router still fails

The two successful local gates do not compose cleanly. In particular:

- The lower high-value gate classifies 2/7 true $500M+ cases as if they belong on the low side because it was not trained to distinguish $500M+ from $50M-$200M.
- The top gate correctly catches 6/7 $500M+ cases but misroutes 1/9 $200M-$500M case upward.
- A simple chain therefore leaves the middle band below its required 8/9 recall once all three bands are present simultaneously.
- A top-first override rescues extreme cases but steals too many lower/middle cases.
- Pairwise score coupling, confidence overrides, and 2-D resolvers were unable to raise the minimum three-band recall above approximately the low-70% range in development screens.

The hard cases are concentrated in the 2020 Biological event. Two middle-band cases illustrate the overlap:

- FEMA 4486, Florida, actual $228.1M: top-boundary score resembles $500M+ cases.
- FEMA 4515, Indiana, actual $213.0M: lower-boundary score resembles $50M-$200M cases.

Within the same event, some lower-funded states have mission/complexity values larger than these middle-band cases, so a simple target-free within-event rank rule is not reliable.

## Additional safe feature families tested and rejected

### Geographic exposure

Complete declaration-scale and county population/land exposure features improved the $50M-$200M side but did not move the $200M-$500M band from 7/9 to 8/9 at both required high boundaries.

### Individual Assistance / IHP nonfinancial counts

Non-dollar IHP counts weakened the relevant high-value boundaries and were rejected.

### Mission composition

Agency, MA type, priority, support-function, authority, and regional composition features, including entropy/HHI/top-share statistics, did not solve the upper boundaries.

### Mission timing

Safe timing/deployment fields improved the $1M-$50M / $50M-$200M boundary but did not solve the top boundaries.

### Public Assistance project structure

The current OpenFEMA `PublicAssistanceFundedProjectsSummaries` source was tested without using `federalObligatedAmount` or any dollar-derived field. Safe features included project counts, applicants, counties, concentration/entropy, and education-applicant share. Raw and event-relative variants failed to solve the high three-way router. Event-relative PA features produced, at best, a $200M-$500M / $500M+ screen of 7/9 vs 6/7; the $50M-$200M / $500M+ comparison remained weak and direct three-class models collapsed the middle-band recall to roughly 33-44%.

### Continuous funding regression

Leave-year-out regression followed by fixed $200M and $500M cutoffs pulled extreme cases toward the dense center and failed badly on $500M+ recall.

### Class balancing / SMOTE

Random oversampling and SMOTE were applied strictly inside each leave-year-out training fold. They did not recover the rare classes. In the tested high-value multiclass configurations, the best minimum recall was approximately 22%, showing that simple imbalance correction does not create the missing separation signal.

### Pre-2010 data expansion

OpenFEMA returned 556 DR declarations for 2000-2009 but no comparable MissionAssignments rows, so no additional historical target support could be constructed. The current MissionAssignments aggregation exactly reproduced the existing master target for 2010-2011, validating the target construction logic but not providing older high-value examples.

## Scientific interpretation

The evidence supports a data-limitation conclusion rather than an algorithm-only conclusion:

1. Each adjacent high boundary can be separated at the required recall when treated locally.
2. The same target-free feature space does not reliably identify all three high bands simultaneously.
3. Multiple unrelated modelling families and additional nonfinancial FEMA sources do not remove the overlap.
4. The rare classes contain only 9 and 7 cases, so one or two mistakes change recall by 11-14 percentage points.
5. Synthetic balancing does not solve the problem, indicating that the limitation is not merely class frequency.

Therefore, with the current 2010-2024 DR master dataset and the tested leakage-safe OpenFEMA feature families, there is not yet evidence for a deployable six-band router that achieves >=80% recall in every final band under strict leave-fiscal-year-out validation.

This is not a claim that such a router is mathematically impossible. It is a documented empirical ceiling for the current dataset and tested target-free feature families. The scientifically clean ways to move beyond the ceiling are to obtain genuinely new high-value observations or a new target-free signal that separates the overlapping high-value cases. The acceptance criterion should not be relaxed or satisfied using target-derived leakage.
