# Research framework

```mermaid
flowchart TD
  D[Immutable FEMA cohort] --> H[Track A historical benchmark]
  D --> J[Geographic and temporal source linkage]
  S[Physical and administrative sources] --> J
  J --> V[Measurement, missingness and release-date audit]
  V --> P[Purged forward obligations estimates]
  P --> U[Chronological uncertainty and review]
  N[Independent estimates of service need] --> A[Simulated budget and allocation comparisons]
  B[Uncertain budget and arrival scenarios] --> A
  H --> R[Human decision briefing]
  U --> R
  V --> R
  A --> R
```

| Module | Output | Evidence standard |
| --- | --- | --- |
| Historical benchmark | Funding bands and amounts | Reproduced frozen historical outputs and artifact provenance |
| Physical evidence | Hazard measurements and source match status | Official measurements, reproducible matching and explicit missingness |
| Forward estimation | Exploratory funding estimates and uncertainty | Purged dates and event groups, earlier-only tuning, conditional coverage |
| Allocation | Feasible service-unit options with equity and delay outcomes | Simulation with independent need, fixed scenarios and explicit weights |
| Human briefing | Evidence status, uncertainty and policy comparisons | Review required for operational interpretation and action |

The original reinforcement learning notebook adjusts base ML recommendations under staged budget releases. Its reward is tied to the archived historical funding outcome. The continuation preserves the staged-budget idea and evaluates service priorities using independent synthetic need. See [original RL notebook](../reinforcement_learning.ipynb), [reproducible human briefing](../analysis/thesis_demo.py).

## Historical funding label eligibility for forward research

Versioned forward candidates must track whether the source could have observed an obligation. The public Mission Assignments v2 history starts October 1, 2012. The 223 earlier recorded zeros are eligible for historical Track A reproduction and excluded from coverage-screened forward training unless a definition-compatible complete ledger can verify them. Later records with absent public actions need a separate completeness audit. A fixed-horizon net obligation and an eventual obligation are distinct forecasting endpoints. [Coverage evidence](continuous/coverage/REPORT.md).
