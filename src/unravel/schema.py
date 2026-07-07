from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jsonschema

REQUIRED_LABELS = (
    "face",
    "eye_l",
    "eye_r",
    "eyebrow_l",
    "eyebrow_r",
    "mouth",
)


class LayersSchemaError(ValueError):
    pass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class SourceImage:
    filename: str
    width: int
    height: int


@dataclass
class Part:
    id: str
    label: str
    layer_file: str
    parent_id: str | None
    depth_order: int
    anchor: Point
    bbox: BBox


@dataclass
class LayersDocument:
    source_image: SourceImage
    parts: list[Part] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source_image": asdict(self.source_image),
            "parts": [
                {
                    "id": p.id,
                    "label": p.label,
                    "layer_file": p.layer_file,
                    "parent_id": p.parent_id,
                    "depth_order": p.depth_order,
                    "anchor": asdict(p.anchor),
                    "bbox": asdict(p.bbox),
                }
                for p in self.parts
            ],
        }


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "layers.schema.json"


def validate_layers_dict(document: dict) -> None:
    schema = json.loads(_schema_path().read_text())
    try:
        jsonschema.validate(instance=document, schema=schema)
    except jsonschema.ValidationError as exc:
        raise LayersSchemaError(str(exc)) from exc

    parts = document["parts"]
    ids = [p["id"] for p in parts]
    if len(ids) != len(set(ids)):
        raise LayersSchemaError(f"duplicate part ids: {ids}")

    for part in parts:
        parent_id = part["parent_id"]
        if parent_id is not None and parent_id not in ids:
            raise LayersSchemaError(
                f"part '{part['id']}' references unknown parent_id '{parent_id}'"
            )

    labels = [p["label"] for p in parts]
    missing = [label for label in REQUIRED_LABELS if label not in labels]
    if missing:
        raise LayersSchemaError(f"missing required part labels: {missing}")

    label_counts = {label: labels.count(label) for label in set(labels)}
    duplicated = [label for label, count in label_counts.items() if count > 1]
    if duplicated:
        raise LayersSchemaError(f"duplicate part labels, expected at most one each: {duplicated}")


def validate_layers_document(document: LayersDocument) -> None:
    validate_layers_dict(document.to_dict())
