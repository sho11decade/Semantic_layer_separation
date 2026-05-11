from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import streamlit as st
from PIL import Image

from semantic_layer_separation.config import load_settings
from semantic_layer_separation.pipeline import _build_layer_relations, run_pipeline
from semantic_layer_separation.exporters.image_export import build_hybrid_alpha, save_alpha, save_cutout, save_mask, save_overlay
from semantic_layer_separation.segmenters.sam2 import SAM2Segmenter, SimpleBoxSegmenter
from semantic_layer_separation.viewer.composition import build_composite_image, resolve_active_layer
from semantic_layer_separation.viewer.loader import LayerSet, discover_layer_sets, load_layer_set


def _load_rgba(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGBA")


def _load_mask(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("L")


def _render_preview(layer_set: LayerSet, visible_indices: set[int], active_index: int | None, mode: str, opacity: float) -> None:
    active_layer = resolve_active_layer(active_index, layer_set.layers)

    image: Image.Image | None = None
    if mode == "Original":
        if layer_set.original_image_path is not None and layer_set.original_image_path.exists():
            image = _load_rgba(layer_set.original_image_path)
        else:
            image = build_composite_image(layer_set, visible_indices, 1.0)
    elif mode == "Composite":
        image = build_composite_image(layer_set, visible_indices, opacity)
    elif active_layer is None:
        st.info("Mask / Cutout / Overlay 表示にはアクティブレイヤーを選択してください。")
        return
    elif mode == "Mask":
        mask = _load_mask(active_layer.mask_path)
        image = None if mask is None else mask
    elif mode == "Cutout":
        image = _load_rgba(active_layer.cutout_path)
    elif mode == "Overlay":
        image = _load_rgba(active_layer.overlay_path)
    elif mode == "Alpha":
        if active_layer.alpha_path is not None:
            image = _load_mask(active_layer.alpha_path)

    if image is None:
        st.warning("表示できる画像がありません。出力ファイルの存在を確認してください。")
        return

    st.image(image, use_container_width=True)


def _render_layer_controls(layer_set: LayerSet) -> set[int]:
    st.sidebar.markdown("### Layers")
    visible_indices: set[int] = set()

    base_key = f"{layer_set.output_dir.resolve()}"
    col1, col2 = st.sidebar.columns(2)
    if col1.button("All ON", use_container_width=True):
        for layer in layer_set.layers:
            st.session_state[f"visible-{base_key}-{layer.index}"] = True
    if col2.button("All OFF", use_container_width=True):
        for layer in layer_set.layers:
            st.session_state[f"visible-{base_key}-{layer.index}"] = False

    for layer in layer_set.layers:
        key = f"visible-{base_key}-{layer.index}"
        if key not in st.session_state:
            st.session_state[key] = True
        checked = st.sidebar.checkbox(
            f"{layer.index:02d} {layer.label}",
            key=key,
        )
        if checked:
            visible_indices.add(layer.index)
    return visible_indices


def _build_viewer_segmenter(settings):
    if settings.sam2_checkpoint and settings.sam2_model_config:
        try:
            return SAM2Segmenter(checkpoint=settings.sam2_checkpoint, model_config=settings.sam2_model_config)
        except Exception:
            return SimpleBoxSegmenter()
    return SimpleBoxSegmenter()


def _load_mask_array(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _default_box_for_layer(layer) -> list[int] | None:
    if layer.box and len(layer.box) == 4:
        return [int(v) for v in layer.box]

    mask = _load_mask_array(layer.mask_path)
    if mask is None or mask.sum() <= 0:
        return None
    ys, xs = np.where(mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max() + 1)
    y1 = int(ys.max() + 1)
    return [x0, y0, x1, y1]


def _clamp_box(box: list[int], *, width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(1, min(width, x1))
    y1 = max(1, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def _append_correction_log(
    *,
    output_dir: Path,
    layer_index: int,
    label: str,
    original_box: list[int] | None,
    corrected_box: list[int],
) -> None:
    corrections_path = output_dir / "corrections.json"
    payload = {"version": "1.0", "corrections": []}
    if corrections_path.exists():
        loaded = _load_json(corrections_path)
        if isinstance(loaded.get("corrections"), list):
            payload["corrections"] = loaded["corrections"]
    payload["corrections"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "layer_index": int(layer_index),
            "label": label,
            "method": "box_correction",
            "original_box": original_box,
            "corrected_box": corrected_box,
        }
    )
    corrections_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_box_correction(layer_set: LayerSet, active_layer, corrected_box: list[int]) -> None:
    if layer_set.original_image_path is None or not layer_set.original_image_path.exists():
        raise FileNotFoundError("Original image is required for correction but was not found.")

    with Image.open(layer_set.original_image_path).convert("RGB") as source_image:
        width, height = source_image.size
    clamped_box = _clamp_box(corrected_box, width=width, height=height)
    if clamped_box is None:
        raise ValueError("Corrected box is invalid after clamping.")

    settings = load_settings()
    segmenter = _build_viewer_segmenter(settings)
    segmented = segmenter.segment(layer_set.original_image_path, [(active_layer.label, tuple(clamped_box))])
    if not segmented:
        raise RuntimeError("Segmentation returned no masks for the corrected box.")

    corrected_mask = np.asarray(segmented[0].mask, dtype=bool)
    alpha_path = active_layer.alpha_path
    if alpha_path is None:
        alpha_name = active_layer.mask_path.name.replace("_mask.png", "_alpha.png")
        alpha_path = active_layer.mask_path.with_name(alpha_name)

    save_mask(corrected_mask, active_layer.mask_path)
    save_cutout(layer_set.original_image_path, corrected_mask, active_layer.cutout_path)
    save_overlay(layer_set.original_image_path, corrected_mask, active_layer.overlay_path)
    save_alpha(build_hybrid_alpha(corrected_mask), alpha_path)

    metadata = _load_json(layer_set.metadata_path)
    layers = metadata.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("Invalid layers.json: 'layers' must be a list.")

    target_entry = None
    for item in layers:
        if isinstance(item, dict) and int(item.get("index", -1)) == int(active_layer.index):
            target_entry = item
            break
    if target_entry is None:
        raise ValueError(f"Layer {active_layer.index} not found in metadata.")

    original_box = None
    box_value = target_entry.get("box")
    if isinstance(box_value, list) and len(box_value) == 4:
        original_box = [int(v) for v in box_value]

    target_entry["box"] = clamped_box
    target_entry["source"] = "manual_correction"
    target_entry["confidence"] = None
    target_entry["alpha_file"] = alpha_path.name

    class _MaskItem:
        def __init__(self, mask: np.ndarray):
            self.mask = mask

    masks = []
    for layer_item in layers:
        if not isinstance(layer_item, dict):
            continue
        mask_name = layer_item.get("mask_file")
        if not isinstance(mask_name, str):
            continue
        mask_array = _load_mask_array(layer_set.output_dir / mask_name)
        if mask_array is None:
            continue
        masks.append(_MaskItem(mask_array))

    if len(masks) == len(layers):
        metadata["relations"] = _build_layer_relations(masks=masks, layers_info=layers)

    layer_set.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_correction_log(
        output_dir=layer_set.output_dir,
        layer_index=active_layer.index,
        label=active_layer.label,
        original_box=original_box,
        corrected_box=clamped_box,
    )


def _render_correction_panel(layer_set: LayerSet, active_index: int | None) -> None:
    st.sidebar.markdown("### Correction")
    active_layer = resolve_active_layer(active_index, layer_set.layers)
    if active_layer is None:
        st.sidebar.caption("補正するには Active layer を選択してください。")
        return

    default_box = _default_box_for_layer(active_layer)
    if default_box is None:
        st.sidebar.warning("現在のレイヤーに有効な box または mask がありません。")
        return

    st.sidebar.caption(f"Target: {active_layer.index:02d} {active_layer.label}")
    x0 = st.sidebar.number_input("x0", min_value=0, value=int(default_box[0]), step=1)
    y0 = st.sidebar.number_input("y0", min_value=0, value=int(default_box[1]), step=1)
    x1 = st.sidebar.number_input("x1", min_value=1, value=int(default_box[2]), step=1)
    y1 = st.sidebar.number_input("y1", min_value=1, value=int(default_box[3]), step=1)

    if st.sidebar.button("Apply box correction", use_container_width=True):
        try:
            _apply_box_correction(layer_set, active_layer, [int(x0), int(y0), int(x1), int(y1)])
            st.sidebar.success("補正を保存しました。")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"補正に失敗しました: {exc}")


def _select_output_dir_mode() -> tuple[Path | None, Path | None]:
    root_input = st.sidebar.text_input("Output root", value="outputs")
    root_dir = Path(root_input).expanduser()

    discovery = discover_layer_sets(root_dir)
    for warning in discovery.warnings:
        st.sidebar.warning(warning)

    if not discovery.layer_dirs:
        return None, None

    selected = st.sidebar.selectbox(
        "Layer output",
        options=discovery.layer_dirs,
        format_func=lambda path: str(path),
    )
    original_input = st.sidebar.text_input("Original image (optional)", value="")
    original_path = Path(original_input).expanduser() if original_input else None
    return selected, original_path


def _run_uploaded_image_mode() -> tuple[Path | None, Path | None]:
    uploaded = st.sidebar.file_uploader("Input image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    prompt = st.sidebar.text_area("Prompt (optional)", value="")
    output_root_input = st.sidebar.text_input("Generated output root", value="outputs")

    generated_dir = st.session_state.get("viewer-generated-dir")
    generated_image = st.session_state.get("viewer-generated-image")

    if st.sidebar.button("Run separation", use_container_width=True):
        if uploaded is None:
            st.sidebar.error("画像ファイルを選択してください。")
            return generated_dir, generated_image

        output_root = Path(output_root_input).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        temp_dir = output_root / "_viewer_inputs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(uploaded.name).suffix or ".png"
        upload_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
        input_path = temp_dir / upload_name
        input_path.write_bytes(uploaded.getbuffer())

        subdir_name = f"viewer_{Path(uploaded.name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            settings = load_settings()
            with st.spinner("レイヤー分離を実行中..."):
                result = run_pipeline(
                    image_path=input_path,
                    settings=settings,
                    prompt=prompt or None,
                    output_subdir=subdir_name,
                )
            original_copy = result.output_dir / f"original{suffix.lower()}"
            shutil.copy2(input_path, original_copy)
            st.session_state["viewer-generated-dir"] = result.output_dir
            st.session_state["viewer-generated-image"] = original_copy
            generated_dir = result.output_dir
            generated_image = original_copy
            st.sidebar.success(f"Generated: {result.output_dir}")
        except Exception as exc:
            st.sidebar.error(f"実行に失敗しました: {exc}")

    return generated_dir, generated_image


def main() -> None:
    st.set_page_config(page_title="Semantic Layer Viewer", layout="wide")
    st.title("Semantic Layer Viewer")

    source_mode = st.sidebar.radio(
        "Input mode",
        options=["Open output directory", "Upload image and run"],
        index=0,
    )

    selected_dir: Path | None = None
    original_image: Path | None = None
    if source_mode == "Open output directory":
        selected_dir, original_image = _select_output_dir_mode()
    else:
        selected_dir, original_image = _run_uploaded_image_mode()

    if selected_dir is None:
        st.info("出力ディレクトリを選択するか、画像をアップロードして実行してください。")
        return

    try:
        layer_set = load_layer_set(selected_dir, original_image_path=original_image)
    except Exception as exc:
        st.error(f"Failed to load layer set: {exc}")
        return

    if not layer_set.layers:
        st.warning("layers.json に有効なレイヤーがありません。")
        return

    for warning in layer_set.warnings:
        st.warning(warning)
    for layer in layer_set.layers:
        for warning in layer.warnings:
            st.warning(f"[Layer {layer.index:02d} {layer.label}] {warning}")

    visible_indices = _render_layer_controls(layer_set)

    st.sidebar.markdown("### View")
    mode = st.sidebar.selectbox(
        "Display mode",
        options=["Original", "Composite", "Mask", "Cutout", "Overlay", "Alpha"],
        index=1,
    )
    opacity = st.sidebar.slider("Composite opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    active_options = {f"{layer.index:02d} {layer.label}": layer.index for layer in layer_set.layers}
    active_label = st.sidebar.selectbox("Active layer", options=["None", *active_options.keys()])
    active_index = None if active_label == "None" else active_options[active_label]
    _render_correction_panel(layer_set, active_index)

    left, right = st.columns([3, 2])
    with left:
        _render_preview(layer_set, visible_indices, active_index, mode, opacity)
    with right:
        st.markdown("### Layer metadata")
        st.caption(f"Output: `{layer_set.output_dir}`  |  Metadata version: `{layer_set.metadata_version}`")
        st.dataframe(
            [
                {
                    "index": layer.index,
                    "label": layer.label,
                    "clean_label": layer.clean_label,
                    "source": layer.source or "",
                    "confidence": layer.confidence,
                    "order_hint": layer.order_hint,
                    "box": layer.box,
                    "material_role": layer.material_role or "",
                    "parent_index": layer.parent_index,
                    "occludes": layer.occludes or [],
                    "visible": layer.index in visible_indices,
                    "mask": layer.mask_path.name,
                    "cutout": layer.cutout_path.name,
                    "overlay": layer.overlay_path.name,
                    "alpha": layer.alpha_path.name if layer.alpha_path is not None else "",
                }
                for layer in layer_set.layers
            ],
            use_container_width=True,
            hide_index=True,
        )
        if layer_set.relations:
            relation_schema = layer_set.relations.get("schema", "unknown")
            parent_edges = layer_set.relations.get("parent_edges", [])
            occlusion_edges = layer_set.relations.get("occlusion_edges", [])
            st.markdown("### Layer relations")
            st.caption(
                f"schema=`{relation_schema}`  |  parent_edges={len(parent_edges)}  |  occlusion_edges={len(occlusion_edges)}"
            )
            if isinstance(parent_edges, list) and parent_edges:
                st.markdown("Parent edges")
                st.dataframe(parent_edges, use_container_width=True, hide_index=True)
            if isinstance(occlusion_edges, list) and occlusion_edges:
                st.markdown("Occlusion edges")
                st.dataframe(occlusion_edges, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
