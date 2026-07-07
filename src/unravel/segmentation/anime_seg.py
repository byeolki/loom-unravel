from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .weights import onnx_model_path

INPUT_SIZE = 1024


class AnimeSegmenter:
    def __init__(self):
        self.session = ort.InferenceSession(
            str(onnx_model_path()), providers=["CPUExecutionProvider"]
        )

    def alpha_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        s = INPUT_SIZE
        normalized = image_rgb.astype(np.float32) / 255.0
        h0, w0 = normalized.shape[:2]
        if h0 > w0:
            h, w = s, max(1, round(s * w0 / h0))
        else:
            h, w = max(1, round(s * h0 / w0)), s
        pad_h, pad_w = s - h, s - w

        canvas = np.zeros((s, s, 3), dtype=np.float32)
        top, left = pad_h // 2, pad_w // 2
        canvas[top : top + h, left : left + w] = cv2.resize(normalized, (w, h))

        tensor = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        mask = self.session.run(["mask"], {"img": tensor})[0][0, 0]
        mask = mask[top : top + h, left : left + w]
        mask = cv2.resize(mask, (w0, h0))
        return np.clip(mask, 0.0, 1.0)
