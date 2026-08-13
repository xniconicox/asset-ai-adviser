# TOPIX近傍500銘柄の個人向け投資分析基盤

日本株を個人で比較・スクリーニングするためのローカル分析基盤です。Yahoo Financeの株価、
J-Quantsの財務履歴、TDnetの適時開示を蓄積し、価格・財務特徴量からRule Rankと6M/12Mの
Ridge Challengerモデルを作成します。自動売買ではなく、候補銘柄の比較と説明を目的としています。

## 記事

実装の背景、データ設計、Point-in-Time学習、評価結果はQiita記事にまとめています。

https://qiita.com/xniconicox009/items/003643f7814ef33e84b0

## まず読む文書

- [PoC README](poc/README.md): セットアップ、取得、学習、画面、日次レポート
- [データ取得方針](docs/データ取得方針.md): J-Quants初回バックフィル、TDnet日次、Yahoo価格
- [TDnet取得設計](docs/TDnet取得設計.md): 原文保存、再実行、LLM構造化
- [学習モデル実装](docs/学習モデル実装.md): 入力、リーク対策、Ridge、評価指標
- [データ品質とクリーニング](docs/データ品質とクリーニング.md): rawを変更しない派生クリーニング
- [定常運用とモデル更新](docs/定常運用とモデル更新.md): 定時取得、レポート、再処理、更新方針
- [リポジトリ構成](docs/リポジトリ構成.md): Git管理対象とローカルデータ

## クイックスタート

    cd poc
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,report]"
    cp .env.example .env
    asset-poc status
    streamlit run src/asset_poc/app.py

環境ファイルには必要に応じてJQUANTS_API_KEY、OPENAI_API_KEYを設定します。APIキーや取得データ、
DuckDB、モデル成果物はGitへ登録しません。

## 主な出力

- 日次ランキングPDF: REPORT_OUTPUT_DIR（既定はOneDriveの共有フォルダ）
- データ充足図PNG: poc/output/coverage/data-coverage-heatmap.png
- 学習表・モデル: poc/output/training/、poc/output/models/

## 現在の制約

現在の学習Universeは最新TOPIX 500相当銘柄を過去へ固定しているため、生存者・構成銘柄バイアスが
残っています。バックテストはCAGRや売買コスト控除後NAVではなく、月次の順位相関と分位リターン
比較です。LLMによるTDnet定性構造化は実装済みですが、定時処理と主Ridgeモデルからは分離しています。
