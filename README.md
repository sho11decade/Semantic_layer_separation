# Semantic_layer_separation
マルチモーダルAIによるセマンティック・レイヤー分離 MVP

## 概要

このリポジトリは、Azure OpenAI でレイヤー候補を抽出し、Grounding DINO で Box を推定し、SAM 2 でマスクを生成して PNG として書き出す研究用 MVP です。

## 実行イメージ

```bash
python -m pip install -e .
cp .env.example .env
sls --image path/to/image.png
```

## 依存関係

Python 3.10 以上を想定しています。`requirements.txt` からインストールしてください。

## 注意

Grounding DINO と SAM 2 はモデル重みと実装依存が必要です。未設定の場合は、まず Azure OpenAI の計画結果と Box 出力までを確認してください。
