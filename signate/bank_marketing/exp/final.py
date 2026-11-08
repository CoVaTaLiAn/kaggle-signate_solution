"""最終提出: 保存済みのテスト予測（各分割モデルの平均）を順位平均して提出ファイルを作る。
先に r03_models.py を実行して preds/ を作っておく。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lib import load, DATA, PRED

NAMES = ["lgb_l15", "lgb_l31_ff5", "cat_d6", "cat_d8"]
te = load()[1]
p = np.mean([rankdata(np.load(PRED / f"{n}_te.npy")) / len(te) for n in NAMES], axis=0)
sub = pd.DataFrame({"id": te["id"], "y": p})
sample = pd.read_csv(DATA / "submit_sample.csv", header=None)
assert (sub["id"].values == sample[0].values).all()
out = DATA.parent / "submit_blend4.csv"
sub.to_csv(out, header=False, index=False, float_format="%.6f")
print(out, sub.shape, sub["y"].describe().to_dict())
