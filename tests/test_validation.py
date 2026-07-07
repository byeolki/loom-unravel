from pathlib import Path

from PIL import Image

import unravel.validation.runner as runner_module
from unravel.pipeline import NoFaceDetectedError
from unravel.schema import BBox, LayersDocument, Part, Point, SourceImage

FULL_LABELS = ("face", "eye_l", "eye_r", "eyebrow_l", "eyebrow_r", "mouth", "hair_front", "hair_back")


def make_document(labels: tuple[str, ...]) -> LayersDocument:
    parts = [
        Part(
            id=label,
            label=label,
            layer_file=f"layers/{label}.png",
            parent_id=None,
            depth_order=0,
            anchor=Point(x=0.0, y=0.0),
            bbox=BBox(x=0, y=0, width=1, height=1),
        )
        for label in labels
    ]
    return LayersDocument(source_image=SourceImage(filename="x.png", width=1, height=1), parts=parts)


def make_input_dir(tmp_path: Path, names: list[str]) -> Path:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for name in names:
        Image.new("RGBA", (10, 10), (255, 255, 255, 255)).save(input_dir / name)
    return input_dir


def test_mixed_success_and_failure(tmp_path, monkeypatch):
    input_dir = make_input_dir(tmp_path, ["good.png", "no_face.png", "partial.png"])

    def fake_run_m1(image_path, output_dir, inpaint=False):
        if image_path.name == "no_face.png":
            raise NoFaceDetectedError("no face")
        if image_path.name == "partial.png":
            return make_document(("face", "eye_l"))
        return make_document(FULL_LABELS)

    monkeypatch.setattr(runner_module, "run_m1", fake_run_m1)

    summary = runner_module.run_validation(input_dir, tmp_path / "out", inpaint=False)

    assert summary.total == 3
    assert summary.parts_pass_count == 1
    assert summary.parts_pass_rate == 1 / 3
    assert not summary.meets_parts_criterion

    by_name = {r.filename: r for r in summary.results}
    assert by_name["good.png"].success
    assert not by_name["no_face.png"].success
    assert by_name["no_face.png"].error == "no face"
    assert not by_name["partial.png"].success
    assert set(by_name["partial.png"].missing_labels) == {"eye_r", "eyebrow_l", "eyebrow_r", "mouth"}


def test_meets_criterion_at_seven_of_ten(tmp_path, monkeypatch):
    names = [f"img_{i}.png" for i in range(10)]
    input_dir = make_input_dir(tmp_path, names)

    def fake_run_m1(image_path, output_dir, inpaint=False):
        index = int(image_path.stem.split("_")[1])
        if index < 7:
            return make_document(FULL_LABELS)
        raise NoFaceDetectedError("no face")

    monkeypatch.setattr(runner_module, "run_m1", fake_run_m1)

    summary = runner_module.run_validation(input_dir, tmp_path / "out", inpaint=False)

    assert summary.parts_pass_rate == 0.7
    assert summary.meets_parts_criterion
