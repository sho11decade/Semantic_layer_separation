from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AzureOpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You analyze an image and return noun-like semantic targets for segmentation from back to front. "
    "Strictly avoid style/color/material words unless they are required to distinguish repeated instances. "
    "Merge near-synonyms into one canonical concept (for example person/human -> character, ground/floor -> ground). "
    "For repeated similar instances, include stable positional qualifiers (left/right/top/bottom/front/back/foreground/background). "
    "Use concise snake_case labels and output JSON only."
)


@dataclass(slots=True)
class PlanningResult:
    targets: list[str]
    raw_text: str


class AzureOpenAIPlanner:
    _TOKEN_ALIASES = {
        "people": "character",
        "person": "character",
        "human": "character",
        "man": "character",
        "woman": "character",
        "girl": "character",
        "boy": "character",
        "pet": "animal",
        "doggo": "dog",
        "kitty": "cat",
        "grounds": "ground",
        "floor": "ground",
        "road": "ground",
        "backdrop": "background",
        "bg": "background",
        "skyline": "sky",
    }
    _PHRASE_ALIASES = {
        "main character": "character",
        "character body": "character_body",
        "foreground character": "foreground_character",
        "background character": "background_character",
    }
    _POSITIONAL_QUALIFIERS = {
        "left",
        "right",
        "top",
        "bottom",
        "upper",
        "lower",
        "front",
        "back",
        "foreground",
        "background",
        "center",
        "middle",
    }

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
                    "Please output a list of concrete noun labels for segmentation from back to front. "
                    "Rules: (1) merge near-synonyms to canonical labels, (2) keep granularity consistent, "
                    "(3) for repeated similar objects include positional qualifiers, "
                    "(4) avoid visual-style words (e.g., red/shiny/cute) unless required for disambiguation, "
                    "(5) use snake_case labels only. Return JSON only with the schema "
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
            normalized = AzureOpenAIPlanner._normalize_target_label(cleaned)
            if not normalized:
                continue
            dedupe_key = " ".join(normalized.replace("_", " ").split())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped_targets.append(normalized)
        return deduped_targets

    @classmethod
    def _normalize_target_label(cls, label: str) -> str:
        canonical = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", label.lower()).replace("-", " ").replace("_", " ")
        canonical = re.sub(r"\s+", " ", canonical).strip()
        if not canonical:
            return ""

        if canonical in cls._PHRASE_ALIASES:
            return cls._PHRASE_ALIASES[canonical]

        tokens = [cls._TOKEN_ALIASES.get(token, token) for token in canonical.split(" ")]
        if not tokens:
            return ""

        if len(tokens) > 1 and tokens[0] in cls._POSITIONAL_QUALIFIERS:
            # Preserve explicit position qualifier for repeated instances.
            normalized = "_".join(tokens[:3])
        else:
            normalized = "_".join(tokens[:2])

        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized
