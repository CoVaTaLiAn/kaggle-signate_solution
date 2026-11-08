# kaggle-signate_solution

Kaggle・SIGNATE のコンペの解法と実験記録。

## コンペ一覧

| コンペ | 課題 | 評価 | 検証スコア | フォルダ |
|---|---|---|---|---|
| SIGNATE Jリーグの観客動員数予測 | 2014年後半戦の観客動員数 | RMSE | 検証 3,099 / 暫定 3,123 | [`signate/jleague_attendance`](signate/jleague_attendance) |
| SIGNATE 銀行の顧客ターゲティング | 定期預金の口座開設の有無 | AUC | CV 0.9420 | [`signate/bank_marketing`](signate/bank_marketing) |

## 共通

- `tools/eda_baseline.py`: CSV の初期分析（クレンジング → 統計 → 特徴量案 → ベースラインモデル → レポート）を行うスクリプト
- コンペのデータ・提出ファイル（CSV）はコミットしない（`.gitignore`）。各フォルダの `data/` に置いて使う
