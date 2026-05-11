from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


@dataclass(slots=True)
class BoundingBox:
    label: str
    score: float
    box: tuple[int, int, int, int]


class GroundingDINODetector:
    def __init__(self, model_name: str, use_cache: bool = True) -> None:
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = GroundingDinoForObjectDetection.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()
        self.use_cache = use_cache
        
        if use_cache:
            from semantic_layer_separation.cache import CacheManager
            self._cache = CacheManager()
        else:
            self._cache = None

    @staticmethod
    def _canonical_label(label: str) -> str:
        return re.sub(r"\s+", " ", label.replace("_", " ").strip().lower())

    @staticmethod
    def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = box
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)

        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _nms(self, boxes: list[BoundingBox], iou_threshold: float) -> list[BoundingBox]:
        ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
        selected: list[BoundingBox] = []
        for candidate in ordered:
            if all(self._iou(candidate.box, keep.box) < iou_threshold for keep in selected):
                selected.append(candidate)
        return selected

    def _post_process_boxes(
        self,
        boxes: list[BoundingBox],
        targets: list[str],
        width: int,
        height: int,
        nms_iou_threshold: float,
        max_per_label: int,
    ) -> list[BoundingBox]:
        target_rank = {self._canonical_label(target): idx for idx, target in enumerate(targets)}
        grouped: dict[str, list[BoundingBox]] = defaultdict(list)

        for box in boxes:
            clamped = self._clamp_box(box.box, width, height)
            if clamped is None:
                continue
            canonical = self._canonical_label(box.label)
            if not canonical:
                continue
            grouped[canonical].append(BoundingBox(label=box.label.strip(), score=box.score, box=clamped))

        filtered: list[BoundingBox] = []
        for label_group in grouped.values():
            kept = self._nms(label_group, nms_iou_threshold)
            if max_per_label > 0:
                kept = kept[:max_per_label]
            filtered.extend(kept)

        return self._sort_boxes_by_target_rank(filtered, target_rank)

    @staticmethod
    def _sort_boxes_by_target_rank(boxes: list[BoundingBox], target_rank: dict[str, int]) -> list[BoundingBox]:
        return sorted(
            boxes,
            key=lambda b: (target_rank.get(GroundingDINODetector._canonical_label(b.label), len(target_rank)), -b.score),
        )

    @staticmethod
    def _extract_boxes(results: list[dict]) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []
        for result in results:
            labels = result.get("text_labels", result.get("labels", []))
            for box, score, label in zip(result["boxes"], result["scores"], labels):
                x0, y0, x1, y1 = [int(value) for value in box.tolist()]
                boxes.append(
                    BoundingBox(
                        label=str(label).strip(),
                        score=float(score),
                        box=(x0, y0, x1, y1),
                    )
                )
        return boxes

    def _merge_two_stage_boxes(
        self,
        *,
        strict_boxes: list[BoundingBox],
        recall_boxes: list[BoundingBox],
        max_per_label: int,
        iou_dedup_threshold: float = 0.85,
    ) -> list[BoundingBox]:
        if max_per_label <= 0:
            return strict_boxes

        merged: list[BoundingBox] = list(strict_boxes)
        per_label_count: dict[str, int] = defaultdict(int)
        for box in strict_boxes:
            per_label_count[self._canonical_label(box.label)] += 1

        for candidate in sorted(recall_boxes, key=lambda b: b.score, reverse=True):
            canonical = self._canonical_label(candidate.label)
            if per_label_count[canonical] >= max_per_label:
                continue

            existing_same_label = [box for box in merged if self._canonical_label(box.label) == canonical]
            if any(self._iou(candidate.box, existing.box) >= iou_dedup_threshold for existing in existing_same_label):
                continue

            merged.append(candidate)
            per_label_count[canonical] += 1

        return merged

    def detect(
        self,
        image_path: Path,
        targets: list[str],
        threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_per_label: int = 1,
    ) -> list[BoundingBox]:
        if not targets:
            return []

        cache_targets = [
            *targets,
            f"__box:{threshold:.4f}",
            f"__text:{text_threshold:.4f}",
            f"__nms:{nms_iou_threshold:.4f}",
            f"__max:{max_per_label}",
        ]
        # Check cache first
        if self._cache:
            cached = self._cache.get_detection_result(image_path, cache_targets)
            if cached:
                return [
                    BoundingBox(label=b["label"], score=b["score"], box=tuple(b["box"]))
                    for b in cached
                ]
        
        image = Image.open(image_path).convert("RGB")
        caption = ". ".join(target.replace("_", " ") for target in targets)
        inputs = self._processor(images=image, text=caption, return_tensors="pt")
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self._device)
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        strict_results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[targets],
        )

        recall_threshold = max(0.05, threshold * 0.7)
        recall_text_threshold = max(0.05, text_threshold * 0.7)
        recall_results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=recall_threshold,
            text_threshold=recall_text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[targets],
        )

        strict_boxes = self._post_process_boxes(
            self._extract_boxes(strict_results),
            targets,
            width=image.width,
            height=image.height,
            nms_iou_threshold=nms_iou_threshold,
            max_per_label=max_per_label,
        )
        recall_boxes = self._post_process_boxes(
            self._extract_boxes(recall_results),
            targets,
            width=image.width,
            height=image.height,
            nms_iou_threshold=min(0.95, nms_iou_threshold + 0.2),
            max_per_label=max(max_per_label * 2, max_per_label + 1),
        )
        boxes = self._merge_two_stage_boxes(
            strict_boxes=strict_boxes,
            recall_boxes=recall_boxes,
            max_per_label=max_per_label,
        )
        target_rank = {self._canonical_label(target): idx for idx, target in enumerate(targets)}
        boxes = self._sort_boxes_by_target_rank(boxes, target_rank)
        
        # Cache result
        if self._cache:
            self._cache.set_detection_result(
                image_path,
                cache_targets,
                [{"label": b.label, "score": b.score, "box": list(b.box)} for b in boxes]
            )
        
        return boxes
