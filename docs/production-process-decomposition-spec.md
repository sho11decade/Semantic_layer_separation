# Production Process Decomposition Specification

このドキュメントは、`semantic-layer-separation` を「1枚絵の製作過程分解」を扱えるプロジェクトへ発展させるための技術仕様を定義する。

## 1. 目的

現行システムは、画像から意味的な対象を抽出し、`mask/cutout/overlay/alpha` と `layers.json` を出力する。

本仕様の目的は、この出力を次の段階へ拡張することである。

1. 意味レイヤー分離 (`semantic layer separation`)
2. 制作工程レイヤー分離 (`production process decomposition`)
3. 対象ごとの親子構造化 (`entity -> process layers`)
4. Viewer 上での工程単位の補正・評価・比較

最終的には、1枚絵を以下の2軸で扱える構造を目指す。

- 意味軸: `character`, `background`, `prop`, `text`, `fx`
- 工程軸: `line_art`, `base_fill`, `shadow`, `highlight`, `effect`, `postprocess`

## 2. 適用範囲

### 2.1 対象画像

- イラスト寄りの静止画
- キャラクター主体または背景付き1枚絵
- 比較的レイヤー分解可能な画風
- 初期段階ではアニメ塗り、ゲーム立ち絵、フラット寄りの塗りを優先

### 2.2 非対象

- 写真の完全な制作工程復元
- 厚塗り・抽象画・フォトバッシュの完全再現
- 作者固有の真の PSD 構造の復元
- 汎用画像編集ソフト互換を初期リリースで完全保証すること

## 3. 用語定義

| 用語 | 定義 |
| --- | --- |
| Semantic layer | 画像中の意味的対象に対応するレイヤー |
| Process layer | 制作工程上の役割に対応するレイヤー |
| Entity | `character` や `desk` などの意味的対象グループ |
| Group | 同一対象に属する複数レイヤーの束 |
| Process role | 制作工程上の役割。例: `line_art`, `shadow` |
| Layer tree | Entity と Process layer を親子関係で表した構造 |
| Editability | 手動修正しやすさ、再利用しやすさを示す性質 |

## 4. 目標システム像

目標とする出力構造の概念図を以下に示す。

```text
Scene
├─ entity: background
│  ├─ process: base_fill
│  ├─ process: shadow
│  └─ process: atmosphere
├─ entity: character_main
│  ├─ process: line_art
│  ├─ process: base_fill
│  ├─ process: shadow
│  ├─ process: highlight
│  └─ process: effect
├─ entity: desk
│  ├─ process: base_fill
│  └─ process: shadow
└─ entity: post_fx
   ├─ process: glow
   └─ process: text
```

この構造では、単一の PNG 群ではなく、意味と工程が明示された編集可能な階層データを生成する。

## 5. 要求事項

### 5.1 機能要求

1. 画像から semantic layer を生成できること
2. semantic layer から process layer を推定できること
3. 各 process layer に `process_role` を付与できること
4. 複数レイヤーを entity 単位にグルーピングできること
5. Layer tree を `layers.json` に保存できること
6. Viewer が process role 単位の表示切替に対応すること
7. Viewer が工程ラベルと親子関係の手動修正に対応すること
8. benchmark が工程分解品質を定量評価できること

### 5.2 非機能要求

- 既存の `layers.json` 互換性を壊さないこと
- SAM2 未設定時も処理継続できること
- 誤判定時に Viewer 補正で救済可能なこと
- 単一画像の処理失敗が batch/benchmark 全体停止につながらないこと
- 中間結果を再利用できること

## 6. 全体アーキテクチャ

### 6.1 処理段階

```text
Input Image
  -> Planning
  -> Detection
  -> Segmentation
  -> Semantic Refinement
  -> Process Classification
  -> Process Decomposition
  -> Relation/Grouping
  -> Export
  -> Viewer / Benchmark
```

### 6.2 段階ごとの責務

| Stage | 責務 | 主な入出力 |
| --- | --- | --- |
| Planning | 分離対象候補の提案 | image -> target labels |
| Detection | bbox 候補生成 | labels -> boxes |
| Segmentation | semantic mask 生成 | boxes -> masks |
| Semantic Refinement | 人物補正、品質ゲート、背景補完 | masks -> refined masks |
| Process Classification | 各レイヤーの工程役割推定 | layer image/mask -> process_role |
| Process Decomposition | 1 entity 内部の工程別サブレイヤー抽出 | entity crop -> process masks |
| Relation/Grouping | 親子、前後関係、グループ構造推定 | layers -> layer tree |
| Export | PNG 群、JSON、補助成果物出力 | layers -> files |

## 7. 提案モジュール構成

以下の新規モジュールを追加し、`pipeline.py` の責務を stage 単位へ分離する。

| パス | 役割 |
| --- | --- |
| `src/semantic_layer_separation/models/layers.py` | 中間データ構造の定義 |
| `src/semantic_layer_separation/stages/planning.py` | planning stage の抽象化 |
| `src/semantic_layer_separation/stages/detection.py` | detection stage の抽象化 |
| `src/semantic_layer_separation/stages/segmentation.py` | segmentation stage の抽象化 |
| `src/semantic_layer_separation/classifiers/process_role_classifier.py` | 工程ラベル推定 |
| `src/semantic_layer_separation/decomposers/process_decomposer.py` | 工程別サブマスク分解 |
| `src/semantic_layer_separation/relations/layer_tree.py` | entity/group/parent/occlusion の構築 |
| `src/semantic_layer_separation/exporters/psd_export.py` | PSD など階層形式への出力 |
| `src/semantic_layer_separation/evaluation/process_metrics.py` | 工程分解の品質指標 |

既存の `providers`, `detectors`, `segmenters`, `refiners`, `viewer` は継続利用しつつ、内部依存を薄くする。

## 8. データモデル

### 8.1 中間データ構造

辞書ベースの中間表現を段階的に dataclass へ置き換える。

```python
SceneResult
  entities: list[EntityLayer]
  relations: SceneRelations

EntityLayer
  id: str
  label: str
  clean_label: str
  entity_type: str
  semantic_mask: np.ndarray
  box: tuple[int, int, int, int] | None
  confidence: float | None
  process_layers: list[ProcessLayer]

ProcessLayer
  id: str
  entity_id: str
  process_role: str
  mask: np.ndarray
  alpha: np.ndarray | None
  blend_hint: str | None
  confidence: float | None
  source: str
```

### 8.2 `layers.json` v2 契約

既存キーを維持したまま拡張する。

ルートキー:

- `version`
- `layers`
- `relations`
- `entities`
- `pipeline_metadata`

各 `layer` の必須キー:

- `index`
- `label`
- `clean_label`
- `mask_file`
- `cutout_file`
- `overlay_file`

各 `layer` の追加キー:

- `entity_id`
- `entity_type`
- `process_role`
- `group_id`
- `blend_hint`
- `stage_confidence`
- `generation_method`
- `semantic_parent_index`
- `process_parent_index`
- `material_role`
- `box`
- `source`
- `confidence`
- `alpha_file`

`entities[]` の例:

```json
{
  "id": "character_main",
  "label": "character_main",
  "entity_type": "character",
  "primary_layer_indices": [3, 4, 5, 6],
  "box": [120, 80, 540, 910]
}
```

`relations` の拡張:

- `schema`: `layer_graph_v2`
- `parent_edges[]`: semantic または process の親子
- `occlusion_edges[]`: 前後関係
- `group_edges[]`: entity と layer の所属関係

### 8.3 後方互換方針

1. 既存の `layers[]` 主体の読み込みを維持する
2. Viewer は `process_role` などの新キーを optional として扱う
3. 旧出力でも Viewer を開けること
4. `version` に応じて loader が解釈を切り替えること

## 9. 工程ラベル体系

### 9.1 初期実装で扱う `process_role`

| role | 意味 | 初期導入優先度 |
| --- | --- | --- |
| `line_art` | 線画、輪郭線、インク相当 | 高 |
| `base_fill` | ベース色、下塗り | 高 |
| `shadow` | 陰影、落ち影、セル影 | 高 |
| `highlight` | ハイライト、縁光、反射強調 | 高 |
| `background` | 背景ベース | 高 |
| `effect` | 発光、粒子、スピード線など | 中 |
| `text` | ロゴ、字幕、文字要素 | 中 |
| `postprocess` | bloom、色収差、全体トーンなど | 中 |
| `rough` | ラフ、下描き | 低 |
| `texture` | パターン、ノイズ、質感付与 | 低 |

### 9.2 `entity_type`

- `character`
- `background`
- `prop`
- `environment`
- `text`
- `fx`
- `unknown`

### 9.3 `blend_hint`

- `normal`
- `multiply`
- `screen`
- `add`
- `overlay`
- `soft_light`
- `unknown`

## 10. 推定パイプライン仕様

### 10.1 Semantic stage

現行 pipeline を土台として維持する。

1. Azure OpenAI で semantic target planning
2. Grounding DINO で bbox 検出
3. SAM2 または矩形 fallback で semantic mask 生成
4. 品質ゲートで低品質マスクを再試行
5. 人物系は Mask R-CNN で再補正
6. 背景残差と描画補完を必要に応じて追加

### 10.2 Process classification stage

semantic layer ごとに `process_role` を推定する。

入力:

- 元画像
- semantic mask
- crop 画像
- 周辺レイヤー情報

特徴量候補:

- 輝度分布
- 彩度分布
- 勾配強度と細線率
- マスク面積比
- 境界近傍の濃度変化
- 周辺レイヤーとの包含関係

推定方式:

1. ルールベース分類器
2. 画像特徴ベースの軽量分類器
3. 将来的に学習ベース分類器へ置換

### 10.3 Process decomposition stage

entity 単位または semantic layer 単位で内部を process layer へ分解する。

初期アルゴリズム案:

- `line_art`
  - エッジ抽出
  - 細線化
  - 高コントラスト領域フィルタ
- `base_fill`
  - semantic mask の内部から高周波を除去した主色面を抽出
- `shadow`
  - semantic mask 内の暗部領域を抽出
  - 局所色差と位置依存を併用
- `highlight`
  - 高輝度かつ局所的な領域を抽出
- `effect`
  - semantic mask 外周や低不透明度疑似領域を residual として抽出

初期方針としては、完全な工程復元ではなく「編集上有用な近似分解」を採用する。

### 10.4 Relation/Grouping stage

以下の関係を推定して保存する。

- Entity 所属関係
- Semantic 親子関係
- Process 親子関係
- Occlusion 関係
- 表示順ヒント

推定ルール例:

- 大きな semantic mask を entity 親とする
- `line_art/base_fill/shadow/highlight` は同一 entity に束ねる
- `shadow/highlight` は `base_fill` を process 親とみなす
- overlap と内包率で `parent_index` を推定する

## 11. Viewer 仕様

### 11.1 必須機能

1. `process_role` でフィルタ表示
2. Entity ごとの折りたたみ表示
3. Semantic view と Process view の切り替え
4. `process_role` の手動修正
5. `entity_type`, `blend_hint` の手動修正
6. 親子関係の手動修正
7. 補正履歴の保存

### 11.2 補正操作

- box 補正
- mask ブラシ補正
- layer の再分類
- group の付け替え
- layer 表示順の変更

### 11.3 保存契約

Viewer の補正結果は `corrections.json` に追記し、可能なら `layers.json` の該当項目も同期更新する。

最低限保存すべき情報:

- `timestamp`
- `layer_index`
- `correction_type`
- `before`
- `after`
- `user_note` 任意

## 12. 評価仕様

### 12.1 評価対象

- semantic mask 品質
- process role 分類精度
- process mask 品質
- group/parent 推定精度
- 編集コスト

### 12.2 指標

| 指標 | 内容 |
| --- | --- |
| IoU | process mask と GT の重なり |
| Boundary F-score | 境界精度 |
| Role accuracy | `process_role` の分類精度 |
| Group consistency | 同一 entity への正しい所属率 |
| Occlusion consistency | 前後関係推定の整合率 |
| Correction cost | 受理までの補正回数 |
| Editability score | 手動編集容易性の複合指標 |

### 12.3 Benchmark report 拡張

`benchmark_report.json` に以下を追加する。

- `avg_process_role_accuracy`
- `avg_process_mask_iou`
- `avg_group_consistency`
- `avg_occlusion_consistency`
- `avg_manual_relabel_count`
- `avg_manual_mask_edit_count`

画像単位の `results[]` にも対応する詳細項目を持たせる。

## 13. データセット仕様

### 13.1 最小評価セット

- 50〜200 枚のイラスト画像
- 画風タグ付き
- 被写体タイプタグ付き
- 工程ラベル付き部分 GT を持つ

### 13.2 アノテーション単位

1. Entity 単位の semantic mask
2. Process role のラベル付け
3. 可能なら process mask の GT
4. 必要に応じて occlusion と group 関係

### 13.3 推奨優先順位

1. `line_art`
2. `base_fill`
3. `shadow`
4. `highlight`
5. `background`

## 14. 設定パラメータ

以下の設定群を追加する。

- `PROCESS_DECOMPOSITION_ENABLED`
- `PROCESS_ROLE_CLASSIFIER`
- `PROCESS_ROLE_CONFIDENCE_THRESHOLD`
- `PROCESS_LINE_ENABLED`
- `PROCESS_BASE_ENABLED`
- `PROCESS_SHADOW_ENABLED`
- `PROCESS_HIGHLIGHT_ENABLED`
- `PROCESS_EFFECT_ENABLED`
- `PROCESS_TREE_ENABLED`
- `PROCESS_EXPORT_PSD_ENABLED`

設計方針:

- semantic 系設定と process 系設定を分離する
- profile ごとに process decomposition のON/OFFを切り替えられること
- 無効時は既存パイプラインと同じ振る舞いになること

## 15. 実装フェーズ

### Phase 1: データ契約拡張

- `layers.json` v2 設計
- loader/viewer の optional field 対応
- benchmark schema 拡張

### Phase 2: 手動アノテーション基盤

- Viewer に `process_role` の手動修正 UI を追加
- correction log を工程情報に対応させる
- 小規模評価セットを構築

### Phase 3: Process role 推定 MVP

- ルールベース分類器追加
- `line_art/base_fill/shadow/highlight/background` を最小対応
- benchmark に role accuracy を追加

### Phase 4: Process decomposition

- entity 内部のサブレイヤー分解
- group/parent relation 強化
- process layer のエクスポート追加

### Phase 5: 編集可能出力

- PSD 出力
- 階層情報付き export
- run 比較と A/B 検証

## 16. 互換性と移行方針

1. 既存 CLI と Viewer のコマンド名は維持する
2. `process decomposition` は feature flag で段階導入する
3. 旧 `layers.json` をそのまま読み込めるようにする
4. `pipeline.py` からの移行中も single/batch/benchmark の表面仕様を変えすぎない
5. export 追加は既存ファイル命名規則を壊さず、補助ファイルとして増やす

## 17. リスクと制約

- 完成絵から真の制作工程を厳密復元することは原理的に困難
- 画風差により工程の定義境界が揺れる
- 厚塗りや写真風では `line_art` や `base_fill` が成立しない場合がある
- semantic 分離誤差が process 分解にも伝播する
- 過剰な自動推定は誤差よりも UI 補正コストを増やす可能性がある

そのため、本プロジェクトでは「真の工程復元」ではなく、「編集上有用な工程近似分解」を基本方針とする。

## 18. 成功条件

以下を満たしたとき、本仕様に対する初期達成とみなす。

1. `process_role` を持つ `layers.json` を出力できる
2. Viewer で工程別に表示・修正できる
3. benchmark で工程ラベル精度を測定できる
4. 少なくとも `line_art/base_fill/shadow/highlight/background` の5分類を自動推定できる
5. 既存の semantic layer separation 機能を破壊していない

## 19. 初期実装の推奨順

1. `layers.json` 拡張
2. Viewer の工程ラベル編集対応
3. ルールベース `process_role` 分類器
4. benchmark 拡張
5. entity 内部の process decomposition
6. PSD など外部編集連携

この順序により、完全自動化に先行して「半自動で回る制作工程分解基盤」を成立させる。
