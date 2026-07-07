from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..pipeline import NoFaceDetectedError, PipelineModels, run_m1
from ..schema import REQUIRED_LABELS, LayersSchemaError
from .report import ImageResult, ValidationSummary

THUMBNAIL_SIZE = 160


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )


def run_validation(input_dir: Path, output_dir: Path, inpaint: bool = False) -> ValidationSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = ValidationSummary()
    models = PipelineModels.build(inpaint=inpaint)

    for image_path in discover_images(input_dir):
        image_output_dir = output_dir / image_path.stem
        try:
            document = run_m1(image_path, image_output_dir, inpaint=inpaint, models=models)
            present_labels = {p.label for p in document.parts}
            missing = [label for label in REQUIRED_LABELS if label not in present_labels]
            summary.results.append(
                ImageResult(
                    filename=image_path.name,
                    success=len(missing) == 0,
                    missing_labels=missing,
                    error=None,
                )
            )
        except (NoFaceDetectedError, LayersSchemaError) as exc:
            summary.results.append(
                ImageResult(
                    filename=image_path.name,
                    success=False,
                    missing_labels=list(REQUIRED_LABELS),
                    error=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001
            summary.results.append(
                ImageResult(
                    filename=image_path.name,
                    success=False,
                    missing_labels=list(REQUIRED_LABELS),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    (output_dir / "validation_report.json").write_text(json.dumps(summary.to_dict(), indent=2))
    build_face_montage(input_dir, output_dir)
    return summary


def load_validation_report(output_dir: Path) -> ValidationSummary:
    data = json.loads((output_dir / "validation_report.json").read_text())
    return ValidationSummary.from_dict(data)


def build_face_montage(input_dir: Path, output_dir: Path) -> Path | None:
    face_paths = sorted(output_dir.glob("*/layers/face.png"))
    if not face_paths:
        return None

    columns = min(5, len(face_paths))
    rows = (len(face_paths) + columns - 1) // columns
    montage = Image.new(
        "RGBA", (columns * THUMBNAIL_SIZE, rows * THUMBNAIL_SIZE), (255, 255, 255, 255)
    )
    for index, face_path in enumerate(face_paths):
        thumbnail = Image.open(face_path).convert("RGBA")
        thumbnail.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        col, row = index % columns, index // columns
        offset = (
            col * THUMBNAIL_SIZE + (THUMBNAIL_SIZE - thumbnail.width) // 2,
            row * THUMBNAIL_SIZE + (THUMBNAIL_SIZE - thumbnail.height) // 2,
        )
        montage.paste(thumbnail, offset, thumbnail)

    montage_path = output_dir / "face_montage.png"
    montage.save(montage_path)
    return montage_path
