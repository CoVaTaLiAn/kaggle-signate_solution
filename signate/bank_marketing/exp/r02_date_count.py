"""日付ごとの集計特徴量（学習+テストを合わせて数える。目的変数は使わない）。"""
import sys
import numpy as np
import pandas as pd
from lib import load, features, cv_predict, lgb_fp, cat_fp, save


def add_date_feats(tr, te, level=3):
    a = pd.concat([tr.assign(_tr=1), te.assign(_tr=0)], ignore_index=True)
    a["date"] = a["month"] + "_" + a["day"].astype(str)
    g = a.groupby("date")
    a["date_cnt"] = g["id"].transform("size")
    if level >= 2:
        a["date_cell_rate"] = g["contact"].transform(lambda s: (s == "cellular").mean())
        a["date_unknown_contact"] = g["contact"].transform(lambda s: (s == "unknown").mean())
        a["date_prev_rate"] = g["pdays"].transform(lambda s: (s != -1).mean())
        a["date_dur_mean"] = g["duration"].transform("mean")
        a["date_campaign_mean"] = g["campaign"].transform("mean")
        a["date_age_mean"] = g["age"].transform("mean")
        a["month_cnt"] = a.groupby("month")["id"].transform("size")
    if level >= 3:
        a["date_contact_cnt"] = a.groupby(["date", "contact"])["id"].transform("size")
        a["dur_rank_in_date"] = g["duration"].rank(pct=True)
        a["campaign_rank_in_date"] = g["campaign"].rank(pct=True)
        a["profile_cnt"] = a.groupby(["age", "job", "marital", "education", "balance"])["id"].transform("size")
    a = a.drop(columns=["date"])
    return a[a._tr == 1].drop(columns="_tr").reset_index(drop=True), \
        a[a._tr == 0].drop(columns=["_tr", "y"]).reset_index(drop=True)


if __name__ == "__main__":
    tr, te = load()
    y = tr["y"]
    for lv in [int(v) for v in (sys.argv[1:] or ["1", "2", "3"])]:
        a, b = add_date_feats(tr, te, lv)
        save(f"lgb_date{lv}", *cv_predict(lgb_fp(), features(a), y, features(b)))
