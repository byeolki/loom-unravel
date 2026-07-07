from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..cache import cache_root

MODEL_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_RESOLUTION = 512
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
NEGATIVE_PROMPT = "blurry, lowres, extra eyes, extra mouth, text, watermark, jpeg artifacts"
SEED = 0


class DiffusionInpainter:
    def __init__(self, device: str | None = None):
        from diffusers import StableDiffusionInpaintPipeline

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_REPO_ID,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            torch_dtype=torch.float32,
            cache_dir=str(cache_root() / "inpainting" / "hf"),
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def inpaint(self, image_rgb: np.ndarray, hole_mask: np.ndarray, prompt: str) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb).resize(
            (MODEL_RESOLUTION, MODEL_RESOLUTION), Image.LANCZOS
        )
        mask = Image.fromarray(hole_mask.astype(np.uint8) * 255).resize(
            (MODEL_RESOLUTION, MODEL_RESOLUTION), Image.NEAREST
        )
        generator = torch.Generator(device=self.device).manual_seed(SEED)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            mask_image=mask,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]
        return np.array(result.resize((width, height), Image.LANCZOS))
