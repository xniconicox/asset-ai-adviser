# TDnet無料閲覧データ取得設計

## 目的と適用範囲

TOPIX Core30・Large70・Mid400相当の対象銘柄について、発表直後の決算短信、
決算説明資料、業績予想・配当予想の修正を収集し、LLM定性構造化の入力原文とする。

JPXの適時開示情報閲覧サービスは、開示と同時に情報を掲載し、開示日を含む31日分を
無料公開している。PDFに加え、決算短信等はXBRLまたはHTMLをダウンロードできる。
一方、公式TDnet APIは有料サービスであるため、PoCでは利用しない。

- TDnet概要: https://www.jpx.co.jp/equities/listing/disclosure/tdnet/index.html
- 無料閲覧サービス: https://www.jpx.co.jp/listing/disclosure/01.html
- 有料TDnet API: https://www.jpx.co.jp/markets/paid-info-listing/tdnet/02.html

無料閲覧ページは機械取得用APIではない。この実装はローカルPoC限定の低頻度
コレクターとし、日付一覧ページを順次取得する。大量バックフィル、並列取得、常時
ポーリング、第三者への再配布には使用しない。ページ仕様変更時には停止して見直す。

## 取得フロー

1. 日付を指定して一覧ページを取得する。無料公開範囲外と未来日は拒否する。
2. 全ページの時刻、5桁コード、会社名、表題、PDF/XBRL URL、取引所、更新履歴を読む。
3. 5桁コード末尾の `0` を除き、PoCの4文字canonical codeへ対応付ける。
4. 現行watchlistを対象に、決算関連表題だけを選択する。
5. PDFと、存在する場合はXBRL ZIPを順次取得する。
6. XBRL ZIPを安全検査後に展開し、`qualitative.htm`またはInline XBRL HTMLを記録する。
7. PDFから本文を抽出して`disclosure_texts`へ登録する。PDFが空の場合はHTMLを使う。
8. LLMは呼ばない。`structure-disclosures`を明示実行した時だけ構造化する。

表題分類は以下を標準対象とする。

| 種別 | 主な判定語 |
|---|---|
| earnings_release | 決算短信 |
| earnings_presentation | 決算説明、決算補足、決算概要 |
| forecast_revision | 業績予想 + 修正・差異・変更 |
| dividend_revision | 配当予想 + 修正・変更 |

決算発表日の変更はスケジュール情報として分類するが、定性分析対象には含めない。
`--scope all`を明示した場合のみ、その他適時開示も取得できる。

## 保存と出典

| 保存先 | 内容 |
|---|---|
| `raw_manifest` | URL、SHA-256、Content-Type、取得時刻、collector version |
| `tdnet_documents` | 一覧メタデータ、原文URL、ローカルパス、状態、本文文字数 |
| `disclosure_texts` | LLMへ渡す抽出本文と原文PDFの出典URL |
| `data/raw/tdnet_list` | 日付別一覧HTML |
| `data/raw/tdnet_pdf` | 原文PDF |
| `data/raw/tdnet_xbrl` | 原文XBRL ZIP |
| `data/raw/tdnet_xbrl_extracted` | XBRL、Inline XBRL、定性HTML |

同一SHA-256のRawは再保存しない。文書IDはPDFファイル名から作るため、同一文書を
再実行しても重複しない。`--force`がなければ取得済み原文はスキップする。

## リカバリと品質管理

- HTTP 429/5xx、接続、読み込み失敗は指数バックオフ付きで最大3回再試行する。
- 文書単位の失敗は他文書を止めず、`retry_queue`へ15分後のpendingとして記録する。
- 同じ日付を再実行すると失敗文書だけを再取得できる。
- 日次処理はwatermark以降を最大7日ずつCatch-upする。無料公開31日より古い欠損は
  回収できないため、公開範囲の最古日に切り上げる。
- `tdnet_document_integrity`はdownloaded文書のPDF、抽出本文、本文登録を検査する。
- `tdnet_pending_retries`は保留文書をWARNとして品質記録へ残す。
- ZIPは絶対パス、`..`、シンボリックリンク、1,000ファイル超、展開後100MB超を拒否する。

## コマンド

```bash
# 対象銘柄の決算関連開示
asset-poc collect-tdnet-date --date 2026-08-11

# 一覧確認のみ
asset-poc collect-tdnet-date --date 2026-08-12 --metadata-only

# 特定銘柄、最大件数を指定
asset-poc collect-tdnet-date --date 2026-08-12 --code 9069 --limit 1

# 対象銘柄の決算以外も含める
asset-poc collect-tdnet-date --date 2026-08-12 --scope all

# 市場全体を対象にする。件数制限を併用する
asset-poc collect-tdnet-date --date 2026-08-12 --market-wide --limit 10
```

`asset-poc daily`はJSTの前日分を自動取得する。LLM構造化は費用管理のため日次処理に
含めず、取得件数と本文を確認後に別コマンドで実行する。

## 実データ確認

2026-08-12にコード9069の決算短信で一覧6ページ・市場526件を解析し、対象1件の
PDF、XBRL ZIP、`qualitative.htm`、10,651文字の本文を取得した。直後の再実行では
原文ダウンロード0件、取得済みスキップ1件となり、冪等性を確認した。
