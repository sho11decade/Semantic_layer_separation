from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from openai import AzureOpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You analyze an image and return only the noun-like semantic targets that should be segmented "
    "from back to front. Output JSON only."
)


@dataclass(slots=True)
class PlanningResult:
    targets: list[str]
    raw_text: str


class AzureOpenAIPlanner:
    def __init__(self, *, api_key: str, endpoint: str, api_version: str, deployment: str, use_cache: bool = True) -> None:
        self._client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self._deployment = deployment
        self.use_cache = use_cache
        
        if use_cache:
            from semantic_layer_separation.cache import CacheManager
            self._cache = CacheManager()
        else:
            self._cache = None

    def plan(self, *, image_path: Path | None = None, prompt: str | None = None, max_targets: int | None = None) -> PlanningResult:
        # Check cache first
        if self._cache and image_path:
            cached = self._cache.get_planning_result(image_path, prompt or "")
            if cached:
                return PlanningResult(targets=cached["targets"], raw_text=cached["raw_text"])
        
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "I want to layer-separate this image as a digital illustration. "
                    "Please output a list of specific nouns that can be used for segmentation, "
                    "considering the stacking order from back to front. Return JSON only with the schema "
                    '{"targets": ["background", "mountains", "character_body", "hair", "sword"]}'
                ),
            },
        ]
        if prompt:
            messages[-1]["content"] = prompt

        content: list[dict[str, object]] = [{"type": "text", "text": str(messages[-1]["content"])}]
        if image_path is not None:
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})

        messages[-1]["content"] = content

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0,
        )
        raw_text = response.choices[0].message.content or ""
        targets = self._extract_targets(raw_text)
        if max_targets is not None and max_targets > 0:
            targets = targets[:max_targets]
        
        # Cache result
        if self._cache and image_path:
            self._cache.set_planning_result(image_path, prompt or "", {"targets": targets, "raw_text": raw_text})
        
        return PlanningResult(targets=targets, raw_text=raw_text)

    @staticmethod
    def _extract_targets(raw_text: str) -> list[str]:
        stripped = raw_text.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return []
            payload = json.loads(stripped[start : end + 1])

        targets = payload.get("targets", []) if isinstance(payload, dict) else []
        if not isinstance(targets, list):
            return []

        deduped_targets: list[str] = []
        seen: set[str] = set()
        for target in targets:
            cleaned = str(target).strip()
            if not cleaned:
                continue
            canonical = " ".join(cleaned.replace("_", " ").lower().split())
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            deduped_targets.append(cleaned)
        return deduped_targets
