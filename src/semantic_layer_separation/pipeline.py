from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semantic_layer_separation.config import Settings


@dataclass(slots=True)
class PipelineResult:
    targets: list[str]
    boxes: list[BoundingBox]
    output_dir: Path


def run_pipeline(*, image_path: Path, settings: Settings, prompt: str | None = None) -> PipelineResult:
    from semantic_layer_separation.detectors.grounding_dino import BoundingBox, GroundingDINODetector
    from semantic_layer_separation.exporters.image_export import ensure_output_dir, save_cutout, save_mask, save_overlay
    from semantic_layer_separation.providers.azure_openai import AzureOpenAIPlanner
    from semantic_layer_separation.segmenters.sam2 import SAM2Segmenter

    output_dir = ensure_output_dir(settings.output_dir)

    planner = AzureOpenAIPlanner(
        api_key=settings.azure_openai_api_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        deployment=settings.azure_openai_deployment,
    )
    planning = planner.plan(image_path=image_path, prompt=prompt)

    detector = GroundingDINODetector(settings.grounding_dino_model)
    boxes = detector.detect(image_path, planning.targets)

    if settings.sam2_checkpoint and settings.sam2_model_config:
        segmenter = SAM2Segmenter(checkpoint=settings.sam2_checkpoint, model_config=settings.sam2_model_config)
        masks = segmenter.segment(image_path, [(box.label, box.box) for box in boxes])
        for index, mask in enumerate(masks, start=1):
            save_mask(mask.mask, output_dir / f"{index:02d}_{mask.label}_mask.png")
            save_cutout(image_path, mask.mask, output_dir / f"{index:02d}_{mask.label}_cutout.png")
            save_overlay(image_path, mask.mask, output_dir / f"{index:02d}_{mask.label}_overlay.png")

    return PipelineResult(targets=planning.targets, boxes=boxes, output_dir=output_dir)