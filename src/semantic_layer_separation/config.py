from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-06-01", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: str = Field(default="gpt-5.4", alias="AZURE_OPENAI_DEPLOYMENT")
    grounding_dino_model: str = Field(default="IDEA-Research/grounding-dino-base", alias="GROUNDING_DINO_MODEL")
    planning_max_targets: int = Field(default=12, ge=1, alias="PLANNING_MAX_TARGETS")
    detection_box_threshold: float = Field(default=0.35, ge=0.0, le=1.0, alias="DETECTION_BOX_THRESHOLD")
    detection_text_threshold: float = Field(default=0.25, ge=0.0, le=1.0, alias="DETECTION_TEXT_THRESHOLD")
    detection_nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0, alias="DETECTION_NMS_IOU_THRESHOLD")
    detection_max_per_label: int = Field(default=3, ge=1, alias="DETECTION_MAX_PER_LABEL")
    background_residual_enabled: bool = Field(default=True, alias="BACKGROUND_RESIDUAL_ENABLED")
    background_residual_min_area_ratio: float = Field(
        default=0.01, ge=0.0, le=1.0, alias="BACKGROUND_RESIDUAL_MIN_AREA_RATIO"
    )
    background_residual_label: str = Field(default="background", alias="BACKGROUND_RESIDUAL_LABEL")
    drawing_completion_enabled: bool = Field(default=False, alias="DRAWING_COMPLETION_ENABLED")
    drawing_completion_base_enabled: bool = Field(default=True, alias="DRAWING_COMPLETION_BASE_ENABLED")
    drawing_completion_shadow_enabled: bool = Field(default=True, alias="DRAWING_COMPLETION_SHADOW_ENABLED")
    drawing_completion_line_enabled: bool = Field(default=True, alias="DRAWING_COMPLETION_LINE_ENABLED")
    drawing_completion_min_area_ratio: float = Field(
        default=0.005, ge=0.0, le=1.0, alias="DRAWING_COMPLETION_MIN_AREA_RATIO"
    )
    drawing_completion_shadow_luma_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, alias="DRAWING_COMPLETION_SHADOW_LUMA_THRESHOLD"
    )
    drawing_completion_edge_quantile: float = Field(
        default=0.88, ge=0.0, le=1.0, alias="DRAWING_COMPLETION_EDGE_QUANTILE"
    )
    sam2_checkpoint: str | None = Field(default=None, alias="SAM2_CHECKPOINT")
    sam2_model_config: str | None = Field(default=None, alias="SAM2_MODEL_CONFIG")
    output_dir: Path = Field(default=Path("outputs"), alias="OUTPUT_DIR")


def load_settings() -> Settings:
    load_dotenv()
    return Settings()
