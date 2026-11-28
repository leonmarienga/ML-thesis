# Continuous Batches 1–4: independent clean replay

Reference commit: `b0b6edb229d112dd6d11896a512bf1112414e4f6`. Run date: 2026-11-28. Local isolated CPython 3.12.14 Linux x86-64 environment with the same 76 dependency versions observed in the prior Actions jobs. Used pinned workflow commands and retained, hash-verified inputs. No new acquisition, no evaluation of FY2025 outcomes and no edit to frozen Track A.

| Batch | Outputs | Result |
| --- | ---: | --- |
| 1: Historical coverage, precoverage ablation and earthquake PGA transfer | 19 | Exact declared generated file sets; retained values agree. |
| 2: Declaration publication and earthquake impacts | 9 | Exact expected file sets; retained values agree. |
| 3: Signed-obligation horizons and water recourse | 10 | Exact expected table sets including compressed histories; retained values agree. |
| 4: Announcement publication and wind verification | 8 | Exact expected file sets; retained values agree. |

Numerical comparisons used relative and absolute tolerances of 1e-6. The water-recourse solver feasibility checks passed for 960 paired policy/scenario evaluations. These are reproduction results, not new data or independent validation of generalization. Follow the pinned [workflow](../../../../.github/workflows/thesis-continuous-batches.yml) for acquisition-free commands. The final combined release replay and real-browser interface evaluation are separate gates.
