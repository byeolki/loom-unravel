from __future__ import annotations

import numpy as np
import torch

from .weights import lama_checkpoint_path

PAD_MODULO = 8


class LamaInpainter:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.jit.load(str(lama_checkpoint_path()), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint(self, image_rgb: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        image_tensor, mask_tensor = self._prepare(image_rgb, hole_mask)

        with torch.inference_mode():
            result = self.model(image_tensor, mask_tensor)

        result = result[0].permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result[:height, :width]

    def _prepare(self, image_rgb: np.ndarray, hole_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        image = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = hole_mask[np.newaxis].astype(np.float32)

        _, height, width = image.shape
        pad_height = self._pad_to(height)
        pad_width = self._pad_to(width)
        image = np.pad(image, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode="symmetric")
        mask = np.pad(mask, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode="symmetric")
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return image_tensor, mask_tensor

    @staticmethod
    def _pad_to(size: int) -> int:
        if size % PAD_MODULO == 0:
            return size
        return (size // PAD_MODULO + 1) * PAD_MODULO
