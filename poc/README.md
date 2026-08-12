# 日本株ランキング開発環境（PoC起点）

このPoCは、株価・決算・定性情報を組み合わせた「ルールベースの相対ランキング」を実装するためのローカル検証環境です。実運用の目標は投資判断を自動執行することではなく、対象銘柄を比較可能なスコア付きの候補群として整理することです。

## 現在の実装の中心

実装の中心は `asset_poc/ranking.py` の `calculate_investment_ranks()` です。これは以下を入力として、6M/12Mの相対順位を出力します。

- `securities`: 銘柄マスタ
- `price_feature_snapshots`: 価格特徴量（モメンタム、ボラティリティ、リスク）
- `fundamental_feature_snapshots`: 財務特徴量（PER、PBR、ROE、成長率など）
- `qualitative_feature_snapshots`: LLMによる定性スコア（任意）

ランキングは機械学習のランキングモデルではなく、ルール重みと相対パーセンタイルで構成されるRule Rankです。スコア化の本体は以下です。

- 6M重み: Earnings 25%, Momentum 30%, Valuation 10%, Quality 10%, Growth 15%, Risk 10%
- 12M重み: Valuation 20%, Quality 20%, Growth 20%, Earnings 20%, Momentum 10%, Risk 10%
- 定性補正: 6M 0.10, 12M 0.08 の信頼度加重補正

定性情報が無い場合は厳密に補正0になる設計です。
加えて、月次Point-in-Timeデータと6M/12Mラベルを生成し、Ridgeによる学習順位をRule Rankと
時系列ホールドアウトで比較できます。

## 実際の入力と処理フロー

```text
JPX universe
  -> watchlist_membership
  -> price_feature_snapshots
      * return_1m / 3m / 6m / 12m
      * momentum_12_1
      * volatility_60d, downside_volatility_60d
      * max_drawdown_252d
      * momentum_score, risk_score
  -> fundamental_feature_snapshots
      * per, pbr, roe
      * operating_margin, sales_yoy, operating_profit_yoy
      * eps_yoy, forecast_eps_revision
      * financial_completeness
  -> optional qualitative_feature_snapshots
      * qualitative_score
      * qualitative_confidence
  -> calculate_investment_ranks()
  -> score_6m / rank_6m / score_12m / rank_12m
```

## 主要データソース

| データ | 取得元 | 料金 | 観点 |
|---|---|---|---|
| 銘柄一覧 | JPX | 無料 | Universe |
| 株価 | Yahoo Finance | 無料 | 価格特徴量 |
| 財務サマリー | J-Quants v2 | Free plan | PER/PBR/ROE等 |
| 開示文書 | TDnet無料閲覧 | 直近31日、キー不要 | 対象銘柄のPDF/XBRL/HTMLと定性分析原文 |
| LLM定性構造化 | OpenAI Responses API等 | 従量課金 | スコア抽出 |

## 重要な制約

- Yahoo Financeは非公式の二次データとして扱い、公式の遅延株価と混在させない
- J-Quantsの決算データは12週間遅延である
- `calculate_fundamental_features()` は比較可能な決算期間を選び、四半期/年度が混ざらないようにしている
- `confidence` は `0.4 + 0.6 * financial_completeness` で算出される
- `positive_reasons` / `negative_reasons` は重み付き寄与度から生成される
- 価格rawは不変で、`price_clean_v1`が異常行の補正・除外と監査イベントを派生生成する
- 無効価格や10暦日超の空白をまたいで長期リターンを計算しない

## 実行

```bash
cd /mnt/c/Users/ShunK/works/asset-ai-adviser/poc
source .venv/bin/activate

# 価格・財務・順位の計算
asset-poc build-watchlist
asset-poc collect-yahoo --period 2y
asset-poc backfill-financials --min-periods 40 --requests-per-minute 30
asset-poc collect-tdnet-date --date 2026-08-11
asset-poc analyze-price-quality  # rawを変更せず品質イベントを再生成
asset-poc compute-features
asset-poc compute-ranks

# 特定銘柄のTDnet取得確認。LLMは呼び出さない
asset-poc collect-tdnet-date --date 2026-08-12 --code 9069 --limit 1

# min-periodsは取得件数ではなく、再取得対象を選ぶ既存開示日数の閾値。
# 対象銘柄ごとにJ-Quantsが返す全履歴を取得する。

# 取得済み本文を明示的にLLM構造化する場合だけ実行
asset-poc structure-disclosures --limit 10

# 原文一致率・token量・人手レビュー欄を評価CSVへ出力
asset-poc evaluate-qualitative

# 月次Point-in-Time学習データを作成（ネットワーク不要）
asset-poc build-training-dataset

# 6M/12M Ridgeを学習し、期間外テストとRule Rank比較を保存
asset-poc train-model --horizon all

# 保存済みの最新モデルを再評価
asset-poc evaluate-model --horizon 6m
asset-poc evaluate-model --horizon 12m

# 画面表示
# 「学習モデル」タブでデータ状況、期間外評価、係数、最新学習順位を確認
streamlit run src/asset_poc/app.py

# テスト
pytest
```

## 参考ドキュメント

- `../docs/運用設計.md`: 日次更新、バックアップ、再試行、パブリッシュ運用
- `../docs/現状と定性情報拡張.md`: 現状実装と定性特徴量の説明
- `../docs/TDnet取得設計.md`: 直近日付の開示取得、保存、再実行、品質管理
- `../docs/学習モデル実装.md`: 学習入力、時系列分割、コマンド、評価指標、制約
- `../docs/定常運用とモデル更新.md`: 日次処理、過去収集、再学習、モデル昇格と復旧
- `../docs/リポジトリ構成.md`: コード、データ、生成成果物の配置とGit管理方針

## 補足

学習モデルは候補順位を検証するためのChallengerです。現在Universe固定の生存者バイアス、売買コスト、業種中立化などが未解消のため、自動売買シグナルとしては扱いません。投資判断に落とす前提として、説明可能性と再現性を優先しています。
