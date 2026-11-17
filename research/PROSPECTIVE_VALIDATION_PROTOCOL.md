# Prospective validation protocol — design only, no new test outcomes

Freeze the code, features and policy before accessing future outcome labels.
The971 historical records and the 59 repeatedly examined flood tests are
exploratory material. Fiscal2025/2026 records are not automatically untouched,
finalized or comparable to the original funding snapshot.

1. Define a new cohort by an outcome-blind rule, preferably declarations issued
   after the frozen protocol date. A prospective FY 2027 cohort beginning after
   this freeze is possible; complete follow-up necessarily requires later data.
   Document partial-year selection. Include all eligible represented hazards,
   and preserve a separately reported non-Biological912-style scope definition.
2. Fix T0 to an authenticated declaration timestamp. If only a date exists,
   report timing ambiguity; do not silently interpret midnight as true issue time.
   Evaluate T+72 separately. Archive original county-designation versions.
3. Prespecify the funding target and follow-up window. A365-day obligation target
   is a possible new estimand, not automatically comparable to eventual funding
   in the historical workbook. Reconstruct the same follow-up target for training
   or explicitly abandon a direct comparison. Do not call obligations “need.”
4. Require both observation time and source availability/publication time no
   later than the decision cutoff. Updated best-track records, final NOAA reports,
   finished mission totals, incident-end dates and retrospective designations
   cannot silently enter a declaration-time predictor.
5. Train on strictly earlier fiscal years. Fit imputation, transforms, feature
   selection and hyperparameters inside earlier-year inner folds. Keep full
   physical-event clusters together, including hurricanes crossing states and
   long flood episodes. Purge overlaps and document boundary events.
6. Predeclare one primary comparison on identical cases and a coverage-inclusive
   analysis on the full eligible cohort. Primary monetary metric: dollar MAE;
   router metrics: balanced accuracy and per-band recall. Also report RMSE,
   R², median error, macro precision/recall/F1, confusion matrices, hazard/year/
   zero/rare-band errors and case influence. Use paired event-cluster uncertainty
   for differences when sample sizes support it.
7. Do not select architectures from outer test outcomes. A bounded candidate
   set may include original features, independently verified physical features,
   verified exposure and combined inputs. Use simpler models unless inner
   validation supports complexity. Preserve every significant candidate version.
8. Reserve earlier disjoint calibration data for80/90% intervals. Assess width
   and empirical coverage by hazard/band where support permits. Under temporal
   drift, report exchangeability limitations. Do not infer rare-band coverage
   from a few catastrophic events or clip intervals to predicted bands.
9. Keep real allocation evaluation separate. Acquire independently justified
   need/vulnerability labels and utility definitions. Fix fairness priorities
   with accountable decision-makers. Until then, scenario results stay synthetic.

No prospective score, finalized target, independent test or operational benefit
is asserted by this protocol. Full completion of this gate requires evidence
not present in the current execution session.
