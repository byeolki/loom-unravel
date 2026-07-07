import pytest

from unravel.schema import BBox, LayersDocument, LayersSchemaError, Part, Point, SourceImage, validate_layers_document

REQUIRED_LABELS = (
    "face",
    "eye_l",
    "eye_r",
    "eyebrow_l",
    "eyebrow_r",
    "mouth",
)


def make_part(label: str, part_id: str | None = None, parent_id: str | None = None) -> Part:
    return Part(
        id=part_id or label,
        label=label,
        layer_file=f"layers/{label}.png",
        parent_id=parent_id,
        depth_order=0,
        anchor=Point(x=1.0, y=2.0),
        bbox=BBox(x=0, y=0, width=10, height=10),
    )


def make_document(parts: list[Part]) -> LayersDocument:
    return LayersDocument(
        source_image=SourceImage(filename="input.png", width=100, height=100),
        parts=parts,
    )


def full_valid_parts() -> list[Part]:
    return [
        make_part("face"),
        make_part("eye_l", parent_id="face"),
        make_part("eye_r", parent_id="face"),
        make_part("eyebrow_l", parent_id="face"),
        make_part("eyebrow_r", parent_id="face"),
        make_part("mouth", parent_id="face"),
        make_part("hair_front"),
        make_part("hair_back"),
    ]


def test_valid_document_passes():
    validate_layers_document(make_document(full_valid_parts()))


def test_missing_required_label_fails():
    parts = [p for p in full_valid_parts() if p.label != "mouth"]
    with pytest.raises(LayersSchemaError, match="missing required part labels"):
        validate_layers_document(make_document(parts))


def test_duplicate_ids_fail():
    parts = full_valid_parts()
    parts.append(make_part("hair_back", part_id="hair_back"))
    with pytest.raises(LayersSchemaError, match="duplicate part ids"):
        validate_layers_document(make_document(parts))


def test_duplicate_labels_fail():
    parts = full_valid_parts()
    parts.append(make_part("face", part_id="face_2"))
    with pytest.raises(LayersSchemaError, match="duplicate part labels"):
        validate_layers_document(make_document(parts))


def test_unknown_parent_id_fails():
    parts = full_valid_parts()
    parts.append(make_part("hair_back", part_id="stray", parent_id="does_not_exist"))
    with pytest.raises(LayersSchemaError, match="unknown parent_id"):
        validate_layers_document(make_document(parts))


def test_invalid_label_rejected_by_json_schema():
    parts = full_valid_parts()
    parts.append(make_part("clothing", part_id="clothing"))
    with pytest.raises(LayersSchemaError):
        validate_layers_document(make_document(parts))


def test_empty_parts_rejected():
    with pytest.raises(LayersSchemaError):
        validate_layers_document(make_document([]))
