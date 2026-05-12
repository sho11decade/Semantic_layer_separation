# 研究的成果サマリ

本研究では、画像のセマンティックレイヤー分離を対象に、Planning（Azure OpenAI）・Detection（Grounding DINO）・Segmentation（SAM2）を統合した実行基盤を構築した。加えて、単発の分離精度だけでなく、後編集での実用性を重視し、(1) ラベル正規化、(2) 2段階検出（recall→precision）、(3) マスク品質ゲート＋再推論、を導入して安定性を向上させた。

新規性として、`layers.json` を拡張し、各レイヤーの属性に加えて `relations`（`layer_graph_v1`）を導入した。これにより、`material_role`・`parent_index`・`occludes` など、編集工程で必要となる構造情報を明示的に扱えるようにした。さらに、Viewer上での手動box補正→再セグメンテーション→履歴保存（`corrections.json`）を実装し、人手を含む反復改善ループをシステムに組み込んだ。

加えて、ハイブリッドalpha出力（`*_alpha.png`）を実装し、硬い2値マスクだけでは扱いにくい境界品質を補完した。評価面では、従来の処理時間・検出数に加え、`uncovered_ratio`、`overlap_conflict_ratio`、`edge_noise_ratio`、`correction_iterations_to_accept` をベンチマーク指標として追加し、編集可能性を定量化する評価設計を整備した。これにより、本システムは「分離結果の生成器」から「編集実務に接続可能な研究基盤」へ拡張された。
    