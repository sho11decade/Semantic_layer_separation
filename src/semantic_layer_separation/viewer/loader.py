from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LayerAsset:
    index: int
    label: str
    clean_label: str
    mask_path: Path
    cutout_path: Path
    overlay_path: Path
    source: str | None = None
    confidence: float | None = None
    order_hint: int | None = None
    box: list[int] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LayerSet:
    output_dir: Path
    metadata_path: Path
    metadata_version: str
    layers: list[LayerAsset]
    original_image_path: Path | None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LayerSetDiscovery:
    layer_dirs: list[Path]
    warnings: list[str] = field(default_factory=list)


def discover_layer_sets(root_dir: Path) -> LayerSetDiscovery:
    root_dir = Path(root_dir).expanduser()
    warnings: list[str] = []
    if not root_dir.exists():
        return LayerSetDiscovery(layer_dirs=[], warnings=[f"Directory not found: {root_dir}"])
    if not root_dir.is_dir():
        return LayerSetDiscovery(layer_dirs=[], warnings=[f"Not a directory: {root_dir}"])

    root_metadata = root_dir / "layers.json"
    if root_metadata.exists():
        return LayerSetDiscovery(layer_dirs=[root_dir], warnings=warnings)

    candidates = sorted(
        child for child in root_dir.iterdir() if child.is_dir() and (child / "layers.json").exists()
    )
    if not candidates:
        warnings.append(f"No layers.json found in: {root_dir}")
    return LayerSetDiscovery(layer_dirs=candidates, warnings=warnings)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("layers.json root must be an object")
    return payload


def _find_original_image(output_dir: Path, explicit_image_path: Path | None) -> Path | None:
    if explicit_image_path is not None:
        candidate = Path(explicit_image_path).expanduser()
        return candidate if candidate.exists() else None

    for name in ("original.png", "original.jpg", "original.jpeg", "source.png", "source.jpg", "source.jpeg"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return None


def load_layer_set(output_dir: Path, *, original_image_path: Path | None = None) -> LayerSet:
    output_dir = Path(output_dir).expanduser()
    metadata_path = output_dir / "layers.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"layers.json not found in {output_dir}")

    payload = _read_json(metadata_path)
    raw_layers = payload.get("layers", [])
    if not isinstance(raw_layers, list):
        raise ValueError("'layers' must be a list")

    layers: list[LayerAsset] = []
    set_warnings: list[str] = []
    for idx, item in enumerate(raw_layers, start=1):
        if not isinstance(item, dict):
            set_warnings.append(f"Layer #{idx} is not an object and was skipped")
            continue

        try:
            index = int(item["index"])
            label = str(item["label"]).strip()
            clean_label = str(item["clean_label"]).strip()
            mask_file = str(item["mask_file"]).strip()
            cutout_file = str(item["cutout_file"]).strip()
            overlay_file = str(item["overlay_file"]).strip()
        except KeyError as exc:
            set_warnings.append(f"Layer #{idx} missing field: {exc}")
            continue
        except (TypeError, ValueError):
            set_warnings.append(f"Layer #{idx} has invalid field types")
            continue

        layer_warnings: list[str] = []
        source = str(item["source"]).strip() if item.get("source") is not None else None
        confidence_raw = item.get("confidence")
        confidence: float | None = None
        if confidence_raw is not None:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                layer_warnings.append(f"Invalid confidence value: {confidence_raw}")

        order_hint_raw = item.get("order_hint")
        order_hint: int | None = None
        if order_hint_raw is not None:
            try:
                order_hint = int(order_hint_raw)
            except (TypeError, ValueError):
                layer_warnings.append(f"Invalid order_hint value: {order_hint_raw}")

        box_raw = item.get("box")
        box: list[int] | None = None
        if box_raw is not None:
            if isinstance(box_raw, list) and len(box_raw) == 4:
                try:
                    box = [int(value) for value in box_raw]
                except (TypeError, ValueError):
                    layer_warnings.append(f"Invalid box values: {box_raw}")
            else:
                layer_warnings.append(f"Invalid box format: {box_raw}")

        mask_path = output_dir / mask_file
        cutout_path = output_dir / cutout_file
        overlay_path = output_dir / overlay_file

        if not mask_path.exists():
            layer_warnings.append(f"Missing mask file: {mask_file}")
        if not cutout_path.exists():
            layer_warnings.append(f"Missing cutout file: {cutout_file}")
        if not overlay_path.exists():
            layer_warnings.append(f"Missing overlay file: {overlay_file}")

        layers.append(
            LayerAsset(
                index=index,
                label=label,
                clean_label=clean_label,
                mask_path=mask_path,
                cutout_path=cutout_path,
                overlay_path=overlay_path,
                source=source,
                confidence=confidence,
                order_hint=order_hint,
                box=box,
                warnings=layer_warnings,
            )
        )

    layers.sort(key=lambda layer: layer.index)
    metadata_version = str(payload.get("version", "unknown"))
    resolved_original = _find_original_image(output_dir, original_image_path)
    if original_image_path is not None and resolved_original is None:
        set_warnings.append(f"Original image not found: {Path(original_image_path).expanduser()}")

    return LayerSet(
        output_dir=output_dir,
        metadata_path=metadata_path,
        metadata_version=metadata_version,
        layers=layers,
        original_image_path=resolved_original,
        warnings=set_warnings,
    )
