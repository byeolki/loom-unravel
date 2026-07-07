from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .cfa_model import INPUT_SIZE, NUM_LANDMARKS, CascadedFaceAlignment
from .weights import cascade_path, checkpoint_path

SCREEN_LEFT_EYE = (0, 10, 11, 12, 13, 14)
SCREEN_RIGHT_EYE = (2, 15, 16, 17, 18, 19)
SCREEN_LEFT_EYEBROW = (3, 4, 5)
SCREEN_RIGHT_EYEBROW = (6, 7, 8)
NOSE = (9,)
MOUTH = (20, 21, 22, 23)
CHIN = (1,)

LANDMARK_GROUPS = {
    "eye_r": SCREEN_LEFT_EYE,
    "eye_l": SCREEN_RIGHT_EYE,
    "eyebrow_r": SCREEN_LEFT_EYEBROW,
    "eyebrow_l": SCREEN_RIGHT_EYEBROW,
    "nose": NOSE,
    "mouth": MOUTH,
    "chin": CHIN,
}


@dataclass
class FaceLandmarks:
    face_bbox: tuple[int, int, int, int]
    points_by_group: dict[str, list[tuple[float, float]]]

    def all_points(self) -> list[tuple[float, float]]:
        return [p for pts in self.points_by_group.values() for p in pts]


class AnimeFaceLandmarkDetector:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.face_detector = cv2.CascadeClassifier(str(cascade_path()))
        self.model = CascadedFaceAlignment(
            output_channel_num=NUM_LANDMARKS + 1,
            checkpoint_path=str(checkpoint_path()),
        )
        self.model.to(self.device)
        self.model.eval()

    def detect_largest_face(self, image_rgb: np.ndarray) -> FaceLandmarks | None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.face_detector.detectMultiScale(gray)
        if len(boxes) == 0:
            return None
        x_, y_, w_, h_ = max(boxes, key=lambda b: b[2] * b[3])
        return self._detect_landmarks(image_rgb, x_, y_, w_, h_)

    def _detect_landmarks(
        self, image_rgb: np.ndarray, x_: int, y_: int, w_: int, h_: int
    ) -> FaceLandmarks:
        height, width = image_rgb.shape[:2]
        x = max(x_ - w_ / 8, 0)
        rx = min(x_ + w_ * 9 / 8, width)
        y = max(y_ - h_ / 4, 0)
        by = min(y_ + h_, height)
        w = rx - x
        h = by - y

        crop = image_rgb[int(y) : int(by), int(x) : int(rx)]
        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        tensor = self._to_tensor(resized).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(tensor)
        heatmaps = heatmaps[-1].cpu().numpy()[0]

        points = []
        for i in range(NUM_LANDMARKS):
            heatmap = cv2.resize(heatmaps[i], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
            ly, lx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            points.append((x + lx * w / INPUT_SIZE, y + ly * h / INPUT_SIZE))

        points_by_group = {
            group: [points[i] for i in indices] for group, indices in LANDMARK_GROUPS.items()
        }
        return FaceLandmarks(
            face_bbox=(int(x), int(y), int(w), int(h)),
            points_by_group=points_by_group,
        )

    @staticmethod
    def _to_tensor(image_rgb_uint8: np.ndarray) -> torch.Tensor:
        normalized = (image_rgb_uint8.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
