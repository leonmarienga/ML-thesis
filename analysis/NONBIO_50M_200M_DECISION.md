# Non-Biological $50M-$200M specialist decision

Strict target-free LFYO audit run: `34764085040`
Artifact: `nonbio-50m-200m-target-free-specialists`
Evaluation cases: 13 non-Biological disasters (`$50M < funding <= $200M`), with 8 in `$50M-$100M` and 5 in `$100M-$200M`.

## Internal router

- Accuracy: **61.54%**
- Balanced accuracy: **61.25%**
- AUC: **0.60**
- Misrouted: **5/13**

The target-free `$50M-$100M` vs `$100M-$200M` handoff is therefore too weak to freeze.

## Funding comparison

| Method | R2 | MAE | RMSE | MedAE | Within 30% |
|---|---:|---:|---:|---:|---:|
| Broad `$50M-$200M` LFYO median | -0.3573 | $35.47M | $48.02M | $21.84M | 46.15% |
| Routed sub-band LFYO medians | -0.3715 | $36.06M | $48.27M | $21.21M | 61.54% |
| Target-free routed preserved specialists | -0.7118 | $42.00M | $53.92M | $27.18M | 53.85% |
| Oracle sub-band LFYO medians | **0.7272** | **$17.07M** | **$21.53M** | **$16.04M** | **84.62%** |
| Oracle preserved specialists | 0.4414 | $23.15M | $30.80M | $17.20M | 76.92% |

The deployable routed specialist stack increases MAE by **18.42%** versus the broad median and by **16.47%** versus routed sub-band medians.

Even with perfect knowledge of the internal sub-band, the simple LFYO sub-band medians outperform the preserved ML specialists (`R2 0.7272` vs `0.4414`, MAE `$17.07M` vs `$23.15M`). This shows that the problem is not only routing; the preserved amount specialists are also too weak for the current 13-case non-Biological high-value subset.

## Decision

**Reject the current `$50M-$100M` / `$100M-$200M` ML specialist stack for final integration.**

For the next end-to-end non-Biological integration, use a training-derived `$50M-$200M` baseline unless a materially better direct amount model is found. Do not report the old oracle fine-split results as deployable performance.

This decision does not alter the frozen six-band router. The broad `$50M-$200M` router performance remains `12/13 = 92.31%`.
