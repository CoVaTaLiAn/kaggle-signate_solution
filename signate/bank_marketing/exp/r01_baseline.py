"""基準: LightGBM（元の特徴量のみ / 追加特徴量あり）、CatBoost。"""
import sys
from lib import load, features, cv_predict, lgb_fp, cat_fp, save

tr, te = load()
y = tr["y"]
which = sys.argv[1:] or ["lgb_raw", "lgb_fe", "cat_fe"]
if "lgb_raw" in which:
    save("lgb_raw", *cv_predict(lgb_fp(), features(tr, False), y, features(te, False)))
if "lgb_fe" in which:
    save("lgb_fe", *cv_predict(lgb_fp(), features(tr), y, features(te)))
if "cat_fe" in which:
    save("cat_fe", *cv_predict(cat_fp(), features(tr), y, features(te)))
