from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


@dataclass(slots=True)
class PersonRefinementResult:
    mask: np.ndarray
    score: float
    box: tuple[int, int, int, int]


def _to_tensor(image_path: Path) -> torch.Tensor:
    with Image.open(image_path).convert("RGB") as image:
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


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
    area_a = max(1, ax1 - ax0) * max(1, ay1 - ay0)
    area_b = max(1, bx1 - bx0) * max(1, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


class PersonMaskRefiner:
    def __init__(
        self,
        *,
        score_threshold: float,
        iou_threshold: float,
        max_instances: int,
    ) -> None:
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._max_instances = max_instances
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights).to(self._device)
        self._model.eval()

    def refine(
        self,
        *,
        image_path: Path,
        target_boxes: list[tuple[int, tuple[int, int, int, int]]],
    ) -> dict[int, PersonRefinementResult]:
        if not target_boxes:
            return {}

        with torch.no_grad():
            tensor = _to_tensor(image_path).to(self._device)
            output = self._model([tensor])[0]

        labels = output.get("labels")
        scores = output.get("scores")
        boxes = output.get("boxes")
        masks = output.get("masks")
        if labels is None or scores is None or boxes is None or masks is None:
            return {}

        candidates: list[PersonRefinementResult] = []
        for label, score, box, mask in zip(labels, scores, boxes, masks):
            if int(label.item()) != 1:  # COCO person class
                continue
            score_value = float(score.item())
            if score_value < self._score_threshold:
                continue
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            mask_bool = np.asarray(mask[0].detach().cpu().numpy() >= 0.5, dtype=bool)
            candidates.append(
                PersonRefinementResult(
                    mask=mask_bool,
                    score=score_value,
                    box=(x0, y0, x1, y1),
                )
            )
            if len(candidates) >= self._max_instances:
                break

        if not candidates:
            return {}

        assigned: set[int] = set()
        results: dict[int, PersonRefinementResult] = {}
        for mask_index, target_box in target_boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, candidate in enumerate(candidates):
                if idx in assigned:
                    continue
                iou = _iou(target_box, candidate.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0 or best_iou < self._iou_threshold:
                continue
            assigned.add(best_idx)
            results[mask_index] = candidates[best_idx]

        return results
