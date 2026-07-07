from __future__ import annotations

import numpy as np

from ..geometry import dilate_mask, ellipse_mask, polygon_mask, rect_mask
from ..landmarks import FaceLandmarks

EYE_MARGIN = 3
EYEBROW_MARGIN = 6
MOUTH_MARGIN = 4
FACE_RADIUS_X_SCALE = 0.95
FACE_RADIUS_Y_SCALE = 1.05
HEAD_REGION_TOP_SCALE = 0.9
HEAD_REGION_SIDE_SCALE = 0.3
HEAD_REGION_BOTTOM_MARGIN_SCALE = 0.15
HAIR_FRONT_MARGIN_SCALE = 0.12


def build_part_masks(
    shape: tuple[int, int], silhouette: np.ndarray, landmarks: FaceLandmarks
) -> dict[str, np.ndarray]:
    fx, fy, fw, fh = landmarks.face_bbox
    center_x, center_y = fx + fw / 2, fy + fh / 2

    face = ellipse_mask(
        shape, center_x, center_y, fw / 2 * FACE_RADIUS_X_SCALE, fh / 2 * FACE_RADIUS_Y_SCALE
    ) & silhouette

    eye_l = polygon_mask(shape, landmarks.points_by_group["eye_l"], margin=EYE_MARGIN) & silhouette
    eye_r = polygon_mask(shape, landmarks.points_by_group["eye_r"], margin=EYE_MARGIN) & silhouette
    eyebrow_l = (
        polygon_mask(shape, landmarks.points_by_group["eyebrow_l"], margin=EYEBROW_MARGIN)
        & silhouette
    )
    eyebrow_r = (
        polygon_mask(shape, landmarks.points_by_group["eyebrow_r"], margin=EYEBROW_MARGIN)
        & silhouette
    )
    mouth = polygon_mask(shape, landmarks.points_by_group["mouth"], margin=MOUTH_MARGIN) & silhouette

    chin_y = landmarks.points_by_group["chin"][0][1]
    head_region = rect_mask(
        shape,
        x0=fx - fw * HEAD_REGION_SIDE_SCALE,
        y0=fy - fh * HEAD_REGION_TOP_SCALE,
        x1=fx + fw * (1 + HEAD_REGION_SIDE_SCALE),
        y1=chin_y + fh * HEAD_REGION_BOTTOM_MARGIN_SCALE,
    )
    hair_total = silhouette & head_region & ~face
    face_near = dilate_mask(face, pixels=round(fw * HAIR_FRONT_MARGIN_SCALE))
    hair_front = hair_total & face_near
    hair_back = hair_total & ~face_near

    return {
        "face": face,
        "eye_l": eye_l,
        "eye_r": eye_r,
        "eyebrow_l": eyebrow_l,
        "eyebrow_r": eyebrow_r,
        "mouth": mouth,
        "hair_front": hair_front,
        "hair_back": hair_back,
    }
