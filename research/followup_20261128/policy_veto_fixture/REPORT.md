# Matched, simulated allocation and veto fixture

This two-dispatch feasibility test compared a **scripted proportional rule**, a **linear optimizer**, and the **optimizer with a scripted veto**, on 48 matched synthetic worlds. It is not a human-subjects study, a trained predictive AI evaluation, or evidence of effective field allocation. The original protocol was hashed before first execution; the study is not externally preregistered.

## Results

| Arm | Overall served | High-vulnerability served | Least-served zone |
|---|---:|---:|---:|
| Scripted equal-fraction rule | 34.69% | 35.15% | 29.54% |
| Optimization alone | 47.51% | 76.05% | 4.46% |
| Optimization with scripted veto | 38.68% | 40.34% | 29.50% |

Compared with optimization alone, the veto arm improved least-served coverage **25.03 percentage points** (paired scenario-resampling interval 20.48 to 29.54), while reducing overall coverage **8.83 points** (−10.50 to −7.16) and high-vulnerability coverage **35.71 points** (−40.07 to −31.33). Compared with the scripted rule, the hybrid arm improved overall coverage **3.99 points** (2.25 to 5.77) and high-vulnerability coverage **5.19 points** (2.76 to 7.77), but observed least-served coverage was very similar (**−0.05 points**, −0.46 to 0.36). This is not evidence of equivalence. The hybrid's least-served share was lower than the rule's in two cases, tied in 43, and higher in three. The rule vetoed the proposal in **45/48** first dispatches and **34/48** second dispatches. Differences reflect the chosen objectives and simulated constraints; the 10-point margin is illustrative and was not tuned after seeing results.

## Design and provenance

The [pinned Batch 14 protocol](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/BATCH14_PROTOCOL.md) supplies the simulator and transport/cargo context. There were 12 new synthetic draws each for response/recovery crossed with the two pinned demand generators, 192 unique seed keys disjoint from the 192 original Batch 14 keys, and 15 normal, 18 delay and 15 closure conditions. They are new random realizations of the *same* generator, not external validation. [CDC USVI CASPER survey](https://stacks.cdc.gov/view/cdc/114255/cdc_114255_DS1.pdf) informs the demand anchors only. Capacities, eight zones, resupply, closures and arrival/handling factors remain modeled. Policies use only the current decision state and carryover at the second dispatch, not second-stage realized states at the first dispatch. The first-stage transport state can be statistically informative of later conditions by construction; later realized capacity, demand, and stocks were not visible before dispatch one.

The heuristic is the archived proportional rule. The optimization-only arm maximizes vulnerability-weighted delivery at the current stage. The veto arm reverts to the proportional rule when the optimization proposal reduces current coverage of the least eligible zone by more than a fixed 10 points. Reachability and stock restrict the veto statistic only; any closed zone with unmet demand stays in the scored denominators. The statistical intervals resample paired synthetic cases within the four design strata (4,000 draws); they do not cover uncertainty over generator assumptions or human behavior and must not be cited as treatment-effect confidence intervals.

## Reproduce

Use the pinned thesis checkout at commit b0b6edb229d112dd6d11896a512bf1112414e4f6, Python 3.12, the pinned research requirements, pdftotext in PATH, and the saved CASPER PDF with SHA-256 9991799f25d8be0aa235048806561674ec97b29126af7b8ebb5a5923f7aa94ba. The source and environment are recorded in results/provenance.json. Use the existing thesis technical-guide retrieval route for the PDF or retrieve the CDC link above and check the hash.

    export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    mkdir -p replay-cache
    export MPLCONFIGDIR="$PWD/replay-cache" XDG_CACHE_HOME="$PWD/replay-cache"
    python replay.py "$REPO" "$SURVEY" replay-out
    python test_decisions.py "$REPO" replay-out
    diff -r results replay-out

The shared run verified 288 scored dispatches against the pinned constraints, held each logged first action unchanged while replacing the later state's roads, checked retained/vetoed proposals and closure denominators, and replayed all five result files byte-for-byte. No FY2025 or obligation data, actual participant behavior, publication, or repository modifications were involved.
