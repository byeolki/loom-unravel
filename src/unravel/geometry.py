from __future__ import annotations

import cv2
import numpy as np


def polygon_mask(
    shape: tuple[int, int], points: list[tuple[float, float]], margin: int = 0
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    if margin > 0:
        kernel = np.ones((margin * 2 + 1, margin * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def ellipse_mask(
    shape: tuple[int, int], center_x: float, center_y: float, radius_x: float, radius_y: float
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (round(center_x), round(center_y)),
        (max(1, round(radius_x)), max(1, round(radius_y))),
        0,
        0,
        360,
        1,
        -1,
    )
    return mask.astype(bool)


def rect_mask(shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    xi0, yi0 = max(0, round(x0)), max(0, round(y0))
    xi1, yi1 = min(width, round(x1)), min(height, round(y1))
    if xi1 > xi0 and yi1 > yi0:
        mask[yi0:yi1, xi0:xi1] = True
    return mask


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def mask_bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    return x0, y0, x1 - x0, y1 - y0


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())
