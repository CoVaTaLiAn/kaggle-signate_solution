"""共通部品: データ読み込み、特徴量、5分割CV（層化）、モデル。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path(__file__).resolve().parent.parent / "data"
PRED = Path(__file__).resolve().parent / "preds"
CATS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, te


def features(df, extra=True):
    X = df.drop(columns=["id", "y"], errors="ignore").copy()
    if extra:
        X["month_num"] = X["month"].map(MONTHS)
        # 年をまたがない近似の「通し日」（月×31 + 日）
        X["doy"] = X["month_num"] * 31 + X["day"]
        X["pdays_none"] = (X["pdays"] == -1).astype(int)
        X["log_duration"] = np.log1p(X["duration"])
        X["dur_per_campaign"] = X["duration"] / X["campaign"]
        X["log_balance"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
    for c in CATS:
        X[c] = X[c].astype("category")
    return X


def cv_predict(fit_predict, X, y, Xte, seeds=(0,), n_splits=5):
    """fit_predict(Xtr, ytr, Xva, yva, Xte, seed) -> (pred_va, pred_te)"""
    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte))
    for s in seeds:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=42 + s)
        for tri, vai in skf.split(X, y):
            pv, pt = fit_predict(X.iloc[tri], y.iloc[tri], X.iloc[vai], y.iloc[vai], Xte, s)
            oof[vai] += pv / len(seeds)
            pte += pt / (n_splits * len(seeds))
    return oof, pte, roc_auc_score(y, oof)


def lgb_fp(params=None, rounds=5000):
    import lightgbm as lgb
    p = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_child_samples=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             verbose=-1, metric="auc")
    p.update(params or {})

    def fp(Xtr, ytr, Xva, yva, Xte, seed):
        m = lgb.train({**p, "seed": seed}, lgb.Dataset(Xtr, ytr), rounds,
                      valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration)
    return fp


def cat_fp(params=None):
    from catboost import CatBoostClassifier

    def fp(Xtr, ytr, Xva, yva, Xte, seed):
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        conv = lambda d: d.astype({c: str for c in cats})
        p = dict(iterations=5000, learning_rate=0.03, depth=6, eval_metric="AUC",
                 random_seed=seed, verbose=0, early_stopping_rounds=300, thread_count=-1)
        p.update(params or {})
        m = CatBoostClassifier(**p, cat_features=cats)
        m.fit(conv(Xtr), ytr, eval_set=(conv(Xva), yva))
        return m.predict_proba(conv(Xva))[:, 1], m.predict_proba(conv(Xte))[:, 1]
    return fp


def save(name, oof, pte, score):
    PRED.mkdir(exist_ok=True)
    np.save(PRED / f"{name}_oof.npy", oof)
    np.save(PRED / f"{name}_te.npy", pte)
    with open(Path(__file__).resolve().parent / "experiments.csv", "a") as f:
        f.write(f"{name},{score:.5f}\n")
    print(f"{name}: CV AUC {score:.5f}", flush=True)
