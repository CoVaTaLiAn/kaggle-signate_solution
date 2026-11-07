# Jリーグの観客動員数予測（SIGNATE）

Jリーグ公式戦 2014年シーズン後半戦（8〜12月）全 313 試合の観客動員数を予測する。評価は RMSE。

## 結果

| | 検証 RMSE |
|---|---|
| 最初の基準（LightGBM、観客数そのまま） | 4,128 |
| **最終モデル（CatBoost 2・Ridge 2・LightGBM の平均）** | **3,099**（暫定スコア 3,123.28） |

- 検証は本番と同じ形: 「2013年7月までで学習 → 2013年8〜12月を予測」（主）と、2012年の同じ形（副）の平均
- 選択に使っていない「前半戦を予測する検証」（2013・2014年の1〜7月）でも 2,920 と安定
- 提出ファイル: `submit_cat2_ridge2_lgb.csv`（`exp/final.py` で作成。CSV はコミットしていない）

## 効いたこと

1. **収容率（観客数 ÷ 収容人数）で学習**し、予測時に収容人数を掛けて戻す（4,128 → 3,735）
2. **試合日時点の順位・勝ち点**（`standings.py`。全試合のスコアから、試合日より前の結果だけで順位表を作る）
3. **CatBoost**: チーム名・審判・スタジアムなどのカテゴリの扱いが上手い
4. **先発のスター選手の人数**（フォルラン、三浦知良、小野伸二など。J2 ではスターがいるチームが来ると収容率 31.7% → 42.4%）
5. **性質の違うモデルの単純平均**（CatBoost・LightGBM・Ridge）
6. **収容人数で重み付け**して、収容率の学習を観客数の RMSE に揃える

効かなかったこと: チームの過去の平均観客数（本番の鮮度に揃えても）、移動距離、日程、log 変換、J1/J2 別モデル、XGBoost、ランダムフォレスト、Optuna によるパラメータ調整（検証への合わせ込みだった）。

詳しい経緯は [`exp/experiments.md`](exp/experiments.md)（全 264 実験、数値は `exp/experiments.csv`）、データの傾向は [`report.md`](report.md)。

## フォルダ構成

| パス | 内容 |
|---|---|
| `data/` | コンペのデータを置く（コミットしない）。`train.csv` `train_add.csv` `test.csv` `2014_add.csv` `condition.csv` `condition_add.csv` `stadium.csv` `sample_submit.csv` |
| `prepare.py` | ファイルの結合と整形（日付・節・中継・天気の分解）→ `data/train_merged.csv` `data/test_merged.csv` |
| `standings.py` | 試合日時点の順位表 → `data/standings.csv` |
| `report.md` / `report.html` / `figs/` / `insights.md` | 初期分析のレポート（データの傾向、特徴量案、考察） |
| `baseline_model.py` | 初期分析のときの初期モデル（検証方法を見直す前のもの） |
| `exp/lib.py` | 実験の共通部品（特徴量、本番と同じ形の検証、モデル、記録） |
| `exp/r01_*.py` 〜 `exp/r21_*.py` | 各ラウンドの実験 |
| `exp/blend.py` | 保存した予測の組み合わせを評価 |
| `exp/final.py` | 最終モデルの学習と提出ファイルの作成 |
| `exp/experiments.md` / `.csv` / `.html` | 全実験の記録 |

## 再現手順

```bash
pip install pandas numpy scipy scikit-learn matplotlib lightgbm catboost xgboost optuna
# data/ にコンペのファイルを置いてから
python3 prepare.py
python3 standings.py
cd exp && python3 final.py cat2_ridge2_lgb   # → ../submit_cat2_ridge2_lgb.csv
```

レポート（`report.md`）は、リポジトリ直下の `tools/eda_baseline.py` で作成した:

```bash
python3 ../../tools/eda_baseline.py run --train data/train_merged.csv --test data/test_merged.csv \
  --target y --id id --time date --metric rmse --out . --title "Jリーグ観客動員数予測"
python3 ../../tools/eda_baseline.py render --out .   # insights.md を反映
```

## 注意

- 2014/3/23 浦和-清水（無観客試合、観客 0 人）は学習から除外した
- `condition` のスコア（試合後の情報）は特徴量に直接は使わず、**それより前の試合の結果から作る順位**にだけ使う
- 検証は2年分しかなく、多くの候補から選んでいるので、本番のスコアは検証より少し悪くなる可能性がある
