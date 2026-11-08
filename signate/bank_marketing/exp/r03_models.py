"""日付集計（level 3）の上で、LightGBM の設定違いと CatBoost を比べる。3シード平均。"""
import sys
from lib import load, features, cv_predict, lgb_fp, cat_fp, save
from r02_date_count import add_date_feats

tr, te = load()
y = tr["y"]
a, b = add_date_feats(tr, te, 3)
X, Xt = features(a), features(b)
S = (0, 1, 2)
runs = {
    "lgb_l15": lambda: lgb_fp(dict(num_leaves=15, min_child_samples=30)),
    "lgb_l63": lambda: lgb_fp(dict(num_leaves=63, min_child_samples=50, learning_rate=0.01)),
    "lgb_l31_ff5": lambda: lgb_fp(dict(feature_fraction=0.5, lambda_l2=5)),
    "cat_d6": lambda: cat_fp(),
    "cat_d8": lambda: cat_fp(dict(depth=8, l2_leaf_reg=5)),
}
for name in (sys.argv[1:] or list(runs)):
    save(name, *cv_predict(runs[name](), X, y, Xt, seeds=S))
