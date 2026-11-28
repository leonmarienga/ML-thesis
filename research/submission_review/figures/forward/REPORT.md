# F7 and T5: matched forward ablation comparison

This audit uses only two published aggregate forward tables from commit `b0b6edb229d112dd6d11896a512bf1112414e4f6`: [confusion matrices](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/results/forward/confusion_matrices.csv) and [paired episode bootstrap](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/results/forward/paired_event_bootstrap.csv). The Python standard-library build checks both SHA-256 inputs, a complete six-by-six matrix for each archived variant, class denominators, alert counts, and the paired dollar contrast. No model was rerun or retuned and no reserved FY2025 outcome was used.

```sh
python -I -S build_f7_t5.py /path/to/pinned/research/results/forward /path/to/out
```

Two clean builds returned byte-identical F7 and T5; the figure was rendered and visually checked at its native 940×546 size. The underlying bootstrap was archived, not re-estimated in this figure build. Exact hits are 192/376 and 224/376; after collapsing bands to a $50M alert the true-positive/false-positive counts are 8/36 and 6/32, respectively. The retrospective-metadata comparator must not be relabeled as the more restrictive metadata-only comparator. The singleton band remains in balanced accuracy. The paired interval crosses zero, so no model is promoted. These data do not support an independently validated model or an operational alert threshold.
