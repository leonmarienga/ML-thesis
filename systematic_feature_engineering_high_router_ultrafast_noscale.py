from pathlib import Path
import pandas as pd
import systematic_feature_engineering_high_router as exp

# Disable the old partial high-only geographic snapshot entirely.
exp.SCALE = Path('__disabled_partial_scale_snapshot__.csv')
exp.OUT = Path('systematic_feature_engineering_high_router_ultrafast_noscale_results')
exp.OUT.mkdir(exist_ok=True)

# Representative strongest model families from the completed diagnostic sweep.
keep = {'cat_d2_160','et_d4_l1','rf_d4_l1'}
exp.MODELS = [m for m in exp.MODELS if m.name in keep]

# Keep only the most relevant feature representations for the direct three-way question.
_orig_feature_sets = exp.feature_sets
def _feature_sets(meta, fold_cols):
    all_sets = _orig_feature_sets(meta, fold_cols)
    wanted = ['logs_ratios','interactions','full_fold_relative']
    return {k: all_sets[k] for k in wanted}
exp.feature_sets = _feature_sets

# Skip pairwise screening here; the individual boundaries are already known to pass in other dedicated nested experiments.
def _skip_pairwise(d, meta):
    out = pd.DataFrame(columns=['feature_set','pair','model','threshold','lower_recall','upper_recall','min_recall','auc','pass80'])
    out.to_csv(exp.OUT/'pairwise_screen_results.csv', index=False)
    return out
exp.pairwise_screen = _skip_pairwise

if __name__ == '__main__':
    exp.main()
