# Technical Architecture

このドキュメントは、`semantic-layer-separation` の技術構成、データ契約、実行フロー、拡張ポイントを整理したものです。

## 1. システム全体像

本プロジェクトは、画像からセマンティックなレイヤーを抽出して PNG 群と `layers.json` を生成するパイプラインです。

1. **Planning**: Azure OpenAI が分離対象ラベルを提案
2. **Detection**: Grounding DINO がラベルごとのバウンディングボックスを推定
3. **Segmentation**: SAM 2 でマスク化（未設定時は矩形マスクへフォールバック）
4. **Export**: `mask/cutout/overlay` と `layers.json` を出力
5. **Completion (optional)**: 背景残差や描画補完レイヤーを追加

## 2. モジュール責務

| モジュール | 責務 |
| --- | --- |
| `src\semantic_layer_separation\cli.py` | 引数処理、設定ロード、single/batch/benchmark の実行分岐 |
| `src\semantic_layer_separation\config.py` | `.env` から設定をロードする契約定義（Pydantic Settings） |
| `src\semantic_layer_separation\pipeline.py` | パイプライン本体、フォールバック、バッチ/ベンチマーク処理 |
| `src\semantic_layer_separation\providers\azure_openai.py` | レイヤー候補の生成 |
| `src\semantic_layer_separation\detectors\grounding_dino.py` | テキスト条件検出と NMS 制御 |
| `src\semantic_layer_separation\segmenters\sam2.py` | SAM2 推論と `SimpleBoxSegmenter` フォールバック |
| `src\semantic_layer_separation\exporters\image_export.py` | 画像/メタデータ出力、ラベルサニタイズ |
| `src\semantic_layer_separation\viewer\` | Streamlit ビューア（読込・合成・表示） |
| `src\semantic_layer_separation\validators.py` | 設定検証と実行前チェック |

## 3. 実行モード

| モード | CLI | 出力 |
| --- | --- | --- |
| Single | `--image` | 1画像分のレイヤー群 + JSON 結果 |
| Batch | `--image-dir` | 画像ごとの出力ディレクトリ + JSON サマリ |
| Benchmark | `--benchmark-dir` | 画像ごとの計測結果 + `benchmark_report.json` |
| Validate | `--validate-config` | 設定妥当性レポート |

## 4. 主要データ契約

### 4.1 `layers.json`

`layers.json` は Viewer と外部ツールの共有契約です。

- ルートキー: `layers`, `version`
- 各 layer 必須キー:
  - `index`
  - `label`
  - `clean_label`
  - `mask_file`
  - `cutout_file`
  - `overlay_file`

命名規則は `NN_clean_label_{mask|cutout|overlay}.png` 形式で、`clean_label` は `sanitize_label` を経由します。

### 4.2 Benchmark Report

`benchmark_report.json` は評価基盤の出力契約です。

- `summary`: 総画像数、成功/失敗数、総処理時間、成功画像の平均時間
- `results[]`: 画像ごとの `duration_ms`, `target_count`, `box_count`, `layer_count`, `status`

## 5. フォールバック設計

このプロジェクトは「失敗時に止めず、劣化して継続する」設計を採用しています。

1. **SAM2 初期化失敗**: `SimpleBoxSegmenter` に切り替え
2. **描画補完/背景補完**: 条件不一致時は追加せず既存レイヤーのみ維持
3. **Batch/Benchmark の個別失敗**: 失敗を記録して次画像へ継続

## 6. 設定パラメータの責務分離

- `PLANNING_*`: LLM の候補生成量
- `DETECTION_*`: 検出品質と重複抑制
- `SAM2_*`: 高精度セグメンテーション有効化
- `BACKGROUND_RESIDUAL_*`: 未カバー領域の背景化
- `DRAWING_COMPLETION_*`: line/base/shadow 補完制御
- `OUTPUT_DIR`: 出力先ルート

これにより、ロジック本体の変更なしで品質・速度・安定性の調整が可能です。

## 7. 拡張方針（実装順）

1. **評価基盤（完了）**: ベンチマークモードで定量評価可能化
2. **検出品質プリセット**: ユースケース別の閾値セット化
3. **キャッシュ統合**: planning/detection の再計算削減
4. **メタデータ拡張**: confidence/source などを追加し互換性維持
5. **バッチレポート強化**: CSV 出力・失敗要約の強化
6. **Viewer 比較機能**: Run A/B 比較を UI で可視化

## 8. 変更時の注意点

- `layers.json` 既存キーは削除しない（Viewer 互換維持）
- SAM2 周辺は「失敗時フォールバック」を壊さない
- 成功時 CLI 出力は機械可読 JSON を維持する
- Windows では `semantic-layer-separation` コマンド表記を優先する
