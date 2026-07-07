from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import mask_bbox, mask_centroid
from .inpainting import LamaInpainter, apply_inpainting
from .landmarks import AnimeFaceLandmarkDetector
from .parts import DEPTH_ORDER, PARENT_OF, build_part_masks
from .schema import BBox, LayersDocument, Part, Point, SourceImage, validate_layers_document
from .segmentation import AnimeSegmenter

LAYER_ORDER = (
    "face",
    "eyebrow_l",
    "eyebrow_r",
    "eye_l",
    "eye_r",
    "mouth",
    "hair_front",
    "hair_back",
)

BBOX_PADDING = 4


class NoFaceDetectedError(RuntimeError):
    pass


@dataclass
class PipelineModels:
    segmenter: AnimeSegmenter
    detector: AnimeFaceLandmarkDetector
    inpainter: LamaInpainter | None = None

    @classmethod
    def build(cls, inpaint: bool = False) -> "PipelineModels":
        return cls(
            segmenter=AnimeSegmenter(),
            detector=AnimeFaceLandmarkDetector(),
            inpainter=LamaInpainter() if inpaint else None,
        )


def run_m1(
    input_path: Path,
    output_dir: Path,
    inpaint: bool = False,
    models: PipelineModels | None = None,
) -> LayersDocument:
    models = models or PipelineModels.build(inpaint=inpaint)

    image = Image.open(input_path).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[:, :, :3]
    base_alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    silhouette = models.segmenter.alpha_mask(rgb) > 0.5

    landmarks = models.detector.detect_largest_face(rgb)
    if landmarks is None:
        raise NoFaceDetectedError(f"no anime face detected in {input_path}")

    masks = build_part_masks(rgb.shape[:2], silhouette, landmarks)

    if inpaint:
        inpainter = models.inpainter or LamaInpainter()
        rgb_by_label = apply_inpainting(rgb, masks, inpainter)
    else:
        rgb_by_label = {label: rgb for label in masks}

    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Part] = []
    for label in LAYER_ORDER:
        mask = masks[label]
        bbox = mask_bbox(mask, padding=BBOX_PADDING)
        if bbox is None:
            continue

        x, y, w, h = bbox
        crop_rgb = rgb_by_label[label][y : y + h, x : x + w]
        crop_mask = mask[y : y + h, x : x + w].astype(np.float32)
        crop_base_alpha = base_alpha[y : y + h, x : x + w]
        alpha = np.clip(crop_mask * crop_base_alpha * 255.0, 0, 255).astype(np.uint8)
        layer_rgba = np.dstack([crop_rgb, alpha])

        layer_filename = f"layers/{label}.png"
        Image.fromarray(layer_rgba, mode="RGBA").save(output_dir / layer_filename)

        anchor_x, anchor_y = mask_centroid(mask)
        parts.append(
            Part(
                id=label,
                label=label,
                layer_file=layer_filename,
                parent_id=PARENT_OF[label],
                depth_order=DEPTH_ORDER[label],
                anchor=Point(x=anchor_x, y=anchor_y),
                bbox=BBox(x=x, y=y, width=w, height=h),
            )
        )

    document = LayersDocument(
        source_image=SourceImage(
            filename=input_path.name, width=rgb.shape[1], height=rgb.shape[0]
        ),
        parts=parts,
    )
    validate_layers_document(document)

    (output_dir / "layers.json").write_text(json.dumps(document.to_dict(), indent=2))
    return document
