"""保存した OOF 予測の順位平均を評価する。python blend.py a b c"""
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from lib import load, PRED

y = load()[0]["y"].values
names = sys.argv[1:]
r = lambda v: rankdata(v) / len(v)
for n in names:
    print(f"{n}: {roc_auc_score(y, np.load(PRED / f'{n}_oof.npy')):.5f}")
oof = np.mean([r(np.load(PRED / f"{n}_oof.npy")) for n in names], axis=0)
print(f"blend({'+'.join(names)}): {roc_auc_score(y, oof):.5f}")
