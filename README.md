# Semantic_layer_separation
マルチモーダルAIによるセマンティック・レイヤー分離 MVP

## 概要

このリポジトリは、Azure OpenAI でレイヤー候補を抽出し、Grounding DINO で Box を推定し、SAM 2（またはシンプルなボックスマスク）でマスクを生成して PNG として書き出す研究用 MVP です。

## パイプライン

1. **Azure OpenAI (GPT-5.4)**: 画像から論理的なセマンティック要素を抽出
2. **Grounding DINO**: テキスト条件付き物体検知で各要素の位置（BBox）を特定
3. **SAM 2 (or SimpleBox)**: BBox からマスクを生成（SAM 2 未設定時は矩形マスク）
4. **PNG 出力**: 各レイヤーの mask、cutout、overlay を保存、`layers.json` で追跡

## 実行方法

```bash
# インストール
python -m pip install -e .

# 設定
cp .env.example .env
# .env を編集: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT を設定

# 実行
sls --image path/to/image.png
```

## 出力ファイル

各層に対して以下の 3 ファイルが生成されます：

- `NN_label_mask.png`: バイナリマスク（グレースケール、0-255）
- `NN_label_cutout.png`: 元画像を alpha チャンネルでマスク（RGBA PNG）
- `NN_label_overlay.png`: 元画像に赤色でハイライト（RGBA PNG）

加えて、`layers.json` でメタデータ（元ラベル、サニタイズ後ラベル、ファイル名対応）を記録します。

## 依存関係

- Python 3.10 以上
- Azure OpenAI API キーとエンドポイント
- `requirements.txt` から pip install（Torch、Transformers、OpenCV、Pillow など）

## 注意

- **SAM 2 は未設定でも動作**: 未設定の場合は `SimpleBoxSegmenter` が矩形マスクを生成
- **Grounding DINO はすぐに使える**: `IDEA-Research/grounding-dino-base` から自動ダウンロード
- **SAM 2 セットアップ** (高精度マスクが必要な場合):
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything-2.git
  ```
  その後、`.env` に `SAM2_CHECKPOINT` と `SAM2_MODEL_CONFIG` を設定
