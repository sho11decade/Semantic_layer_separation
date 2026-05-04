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
    sam2_checkpoint: str | None = Field(default=None, alias="SAM2_CHECKPOINT")
    sam2_model_config: str | None = Field(default=None, alias="SAM2_MODEL_CONFIG")
    output_dir: Path = Field(default=Path("outputs"), alias="OUTPUT_DIR")


def load_settings() -> Settings:
    load_dotenv()
    return Settings()