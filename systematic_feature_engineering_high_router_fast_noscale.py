from pathlib import Path
import systematic_feature_engineering_high_router as exp

# Scientifically clean fast diagnostic: disable the old partial declaration-scale snapshot.
exp.SCALE = Path('__disabled_partial_scale_snapshot__.csv')

# Keep a representative model family set so this finishes quickly while the full sweep runs.
keep = {'cat_d2_160','et_d4_l1','logit_c1'}
exp.MODELS = [m for m in exp.MODELS if m.name in keep]
exp.PAIR_MODELS = [m for m in exp.PAIR_MODELS if m.name in keep]

# Exclude the redundant numeric-only feature-set in this fast screen.
_orig_feature_sets = exp.feature_sets
def _feature_sets(meta, fold_cols):
    out = _orig_feature_sets(meta, fold_cols)
    out.pop('full_numeric_no_cats', None)
    return out
exp.feature_sets = _feature_sets

# Separate outputs from the full sweep.
exp.OUT = Path('systematic_feature_engineering_high_router_fast_noscale_results')
exp.OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    exp.main()
