# Semantic_layer_separation
マルチモーダルAIによるセマンティック・レイヤー分離 MVP

## 概要

このリポジトリは、Azure OpenAI でレイヤー候補を抽出し、Grounding DINO で Box を推定し、SAM 2（またはシンプルなボックスマスク）でマスクを生成して PNG として書き出す研究用 MVP です。

## 技術ドキュメント

- [Technical Architecture](docs/technical-architecture.md)

## パイプライン

1. **Azure OpenAI (GPT-5.4)**: 画像から論理的なセマンティック要素を抽出
2. **Grounding DINO**: テキスト条件付き物体検知で各要素の位置（BBox）を特定
3. **SAM 2 (or SimpleBox)**: BBox からマスクを生成（SAM 2 未設定時は矩形マスク）
4. **PNG 出力**: 各レイヤーの mask、cutout、overlay を保存、`layers.json` で追跡

## プロジェクト構成

```text
Semantic_layer_separation/
├─ src/
│  └─ semantic_layer_separation/
│     ├─ cli.py                  # CLI エントリポイント（sls）
│     ├─ pipeline.py             # 単体/バッチ処理の主フロー
│     ├─ config.py               # 環境変数・設定ロード
│     ├─ validators.py           # 設定バリデーション
│     ├─ errors.py               # 例外定義
│     ├─ logging_config.py       # ログ設定
│     ├─ providers/
│     │  └─ azure_openai.py      # レイヤー候補生成
│     ├─ detectors/
│     │  └─ grounding_dino.py    # BBox 検出
│     ├─ segmenters/
│     │  └─ sam2.py              # SAM 2 / 矩形マスク
│     └─ exporters/
│        ├─ image_export.py      # mask/cutout/overlay 保存
│        └─ archive_export.py    # アーカイブ出力
├─ requirements.txt              # 依存関係（pip）
└─ pyproject.toml                # パッケージ設定
```

## 実行方法

### macOS / Linux

```bash
# インストール
python -m pip install -e .

# 設定
cp .env.example .env
# .env を編集: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT を設定

# 実行
sls --image path/to/image.png
```

### Windows (PowerShell)

```powershell
# インストール
python -m pip install -e .

# 設定
Copy-Item .env.example .env
# .env を編集: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT を設定

# 実行（PowerShell の `sls` エイリアス衝突を回避）
semantic-layer-separation --image .\path\to\image.png
# または
python -m semantic_layer_separation.cli --image .\path\to\image.png

# プロファイル指定（用途別プリセット）
semantic-layer-separation --image .\path\to\image.png --profile illustration

# ベンチマーク実行（固定画像セットの評価）
semantic-layer-separation --benchmark-dir .\path\to\images
# レポート出力先を指定する場合
semantic-layer-separation --benchmark-dir .\path\to\images --benchmark-report .\outputs\benchmark_report.json
```

## レイヤービューア（Streamlit）

```bash
# 依存を反映
python -m pip install -e .

# ビューア起動
semantic-layer-viewer
```

ビューアは2モードをサポートします。

- **Open output directory**: 既存の `outputs/`（またはその配下）を読み込み
- **Upload image and run**: 画像をアップロードしてパイプライン実行後、結果を即時表示

主なUI操作:

- レイヤーの表示ON/OFF
- アクティブレイヤー選択
- 表示モード切替（Original / Composite / Mask / Cutout / Overlay）
- Composite不透明度調整

## 出力ファイル

各層に対して以下の 3 ファイルが生成されます：

- `NN_label_mask.png`: バイナリマスク（グレースケール、0-255）
- `NN_label_cutout.png`: 元画像を alpha チャンネルでマスク（RGBA PNG）
- `NN_label_overlay.png`: 元画像に赤色でハイライト（RGBA PNG）

加えて、`layers.json` でメタデータ（元ラベル、サニタイズ後ラベル、ファイル名対応）を記録します。

ベンチマーク実行時は、処理時間・ターゲット数・ボックス数・レイヤー数を含む集計 JSON（既定: `outputs/benchmark_report.json`）も出力されます。

`layers.json` の各レイヤーは次のキーを前提にします:

- `index`
- `label`
- `clean_label`
- `mask_file`
- `cutout_file`
- `overlay_file`

加えて、次の任意キーが出力されます（後方互換維持）:

- `source`: レイヤー生成元（`detector_segmenter` / `drawing_completion` / `background_residual`）
- `confidence`: 検出信頼度（推定不可のレイヤーでは `null`）
- `order_hint`: 検出順ベースの順序ヒント（推定不可では `null`）
- `box`: 検出BBox `[x0, y0, x1, y1]`（非検出レイヤーでは `null`）
- `material_role`: レイヤーの役割（`background` / `object` / `line_art` / `shadow` など）
- `parent_index`: 推定親レイヤーの index（推定不可では `null`）
- `occludes`: このレイヤーが手前で重なる背面レイヤー index の配列

また、ルートキーとして任意の `relations` が出力され、`parent_edges` / `occlusion_edges` に構造化関係を保持します。

## 依存関係

- Python 3.10 以上
- Azure OpenAI API キーとエンドポイント
- `requirements.txt` から pip install（Torch、Transformers、OpenCV、Pillow など）

## 注意

- **SAM 2 は未設定でも動作**: 未設定の場合は `SimpleBoxSegmenter` が矩形マスクを生成
- **Grounding DINO はすぐに使える**: `IDEA-Research/grounding-dino-base` から自動ダウンロード
- **品質調整パラメータ**: `.env` で `PLANNING_MAX_TARGETS`、`DETECTION_BOX_THRESHOLD`、`DETECTION_TEXT_THRESHOLD`、`DETECTION_NMS_IOU_THRESHOLD`、`DETECTION_MAX_PER_LABEL`、`BACKGROUND_RESIDUAL_ENABLED`、`BACKGROUND_RESIDUAL_MIN_AREA_RATIO`、`BACKGROUND_RESIDUAL_LABEL`、`DRAWING_COMPLETION_*` を調整可能
- **Planning ラベル正規化**: LLM出力は同義語統合・snake_case正規化を行い、重複/揺れを抑制
- **2段階検出**: Recall重視（低閾値）→Precision重視（既定閾値）を統合して取りこぼしを削減
- **マスク品質ゲート**: 面積比/ボックス充填率/境界接触率で低品質マスクを判定し、box拡縮で再推論
- **用途別プロファイル**: `--profile default|illustration|product` で検出閾値・補完レイヤー設定を一括切替可能
- **同一/近似物体の扱い**: 既定で `DETECTION_MAX_PER_LABEL=3` とし、同ラベル物体を複数保持。さらに Planning プロンプトで位置修飾（left/right, front/back など）を促し、近似物体を区別しやすくしています。
- **背景の補完レイヤー**: 物体として検出されなかった領域は、既存マスクの未カバー領域から残差背景レイヤーとして追加可能（既定有効）
- **描画手順ベース補完レイヤー**: `DRAWING_COMPLETION_ENABLED=true` で、既存マスク群から線画（line_art）・下塗り（base_fill）・影（shadow）をルール補完して追加可能
- **SAM 2 セットアップ** (高精度マスクが必要な場合):
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything-2.git
  ```
  その後、`.env` に `SAM2_CHECKPOINT` と `SAM2_MODEL_CONFIG` を設定
- **パスは実在ファイルを指す必要あり**: たとえば `SAM2_MODEL_CONFIG=./sam2/configs/models/sam2.1_hiera_l.yaml` は、リポジトリ直下に `sam2/` を clone している場合のみ有効です。存在しないパスを入れると、SAM 2 初期化は失敗し、矩形マスクにフォールバックします。
