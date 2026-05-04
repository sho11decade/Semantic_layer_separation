from __future__ import annotations

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
    def __init__(self, model_name: str) -> None:
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = GroundingDinoForObjectDetection.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    def detect(self, image_path: Path, targets: list[str], threshold: float = 0.35, text_threshold: float = 0.25) -> list[BoundingBox]:
        image = Image.open(image_path).convert("RGB")
        caption = ". ".join(targets)
        inputs = self._processor(images=image, text=caption, return_tensors="pt")
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self._device)
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[targets],
        )
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