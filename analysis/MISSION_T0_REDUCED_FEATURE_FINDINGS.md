# Mission action T0/T72 and reduced-feature forward-year comparison

Successful runs:
- Mission audit: https://github.com/leonmarienga/ML-thesis/actions/runs/41340860069
- Reduced feature experiment: https://github.com/leonmarienga/ML-thesis/actions/runs/41341023238

## Source-timing audit
Current OpenFEMA MissionAssignments v2 returned 29,084 action rows associated with 452 of the 971 FEMA records. Every action had a dateReceived field. Only 217 actions had dateReceived on/before the **midnight UTC declaration-date cutoff**, covering **48 of 971 disasters**; 2,283 actions were eligible by T+72h, covering 251 of 971 disasters. Among 912 non-Biological records, only **36** had missions on/before T0.

**Critical limitation:** 29,084/29,084 current action records had `lastRefresh` after the T0 cutoff. The extraction filters action creation timestamps but **cannot reconstruct exact historical versions of field values from a current API snapshot**. These are timestamp-screened reconstructed proxies, not fully verified prospective records.

## Experimental forward-year amount comparison

Experimental restricted-feature five-nearest specialist, training on earlier fiscal years only, evaluated 9 true $50–200M cases and 10 cases routed to that band.

| Scope | n | Predictor variant | MAE ($M) | RMSE ($M) | Within ±30% |
|---|---:|---|---:|---:|---:|
| Direct true band | 9 | Metadata (state, incident type, fiscal year, declaration delay) | 51.199 | 60.910 | 33.33% |
| Direct true band | 9 | Metadata + T0 mission proxies | 51.854 | 61.308 | 44.44% |
| Direct true band | 9 | Metadata + T0 mission proxies + recent NHC (≤72h) | **48.176** | 60.752 | 44.44% |
| Frozen-router routed | 10 | Metadata | 60.945 | 74.492 | 30.00% |
| Frozen-router routed | 10 | Metadata + T0 mission proxies | 61.271 | 74.264 | 40.00% |
| Frozen-router routed | 10 | Metadata + T0 mission proxies + recent NHC | **60.328** | 78.750 | 40.00% |

The newest configuration improves direct MAE by ~5.90% against metadata-only and routed MAE by ~1.01%, but routed RMSE worsens. Mission features alone do **not** improve MAE. This demonstrates neither statistically robust improvement nor reliable operational performance.

### Important comparability constraint
These are new lower-information research models. Do NOT compare them as if they were the accepted model: the accepted $5.015M system uses eventual mission operational features and differs in architecture/feature availability. Accurate comparison requires common information cutoffs and an untouched future holdout.

### Next
1. Verify FEMA original declaration versions and as-of issuance timestamps, not only current OpenFEMA records.
2. Obtain versioned mission-assignment snapshots where possible; current lastRefresh makes reconstruction inherently uncertain.
3. Add immutable early warning/weather source features from independent archives for all hazards.
4. Expand the evaluation population beyond 9/10 rare-band cases and report uncertainty before deployment.
5. Do not change frozen six-band router (830/912) or accepted specialists based on this result.
