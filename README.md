# Unravel

Unravel decomposes a single character illustration into semantic, RGBA face-part layers plus a metadata file describing their hierarchy, depth order, and anchor points. It is the first stage of the [Loom](../PLAN.md) pipeline:

```
illustration -> Unravel -> layers/*.png + layers.json -> Warp -> rig.json -> Strings -> parameters -> Frame -> render
```

No custom model is trained. Unravel wires together existing pretrained models — an anime face detector, a landmark regressor, a character silhouette segmenter, and an image inpainter — into a single offline batch pipeline.

## Scope

- Front-facing, single-character, upper-body illustrations only.
- Face-region parts only: `face`, `eye_l`, `eye_r`, `eyebrow_l`, `eyebrow_r`, `mouth`, `hair_front`, `hair_back`.
- Full-body parts, multiple characters, and non-frontal poses are out of scope for now.

## Pipeline

| Stage | Model | Package |
|---|---|---|
| Face + landmark detection | `lbpcascade_animeface` + a VGG16-backboned CFA landmark regressor (24 points) | `unravel.landmarks` |
| Character silhouette | `skytnt/anime-seg` (ISNet, ONNX) | `unravel.segmentation` |
| Part mask construction | Rule-based, built from landmark groups + the silhouette | `unravel.parts` |
| Occlusion inpainting | LaMa (`big-lama.pt`, TorchScript) | `unravel.inpainting` |
| Output contract | `layers.json`, validated against `schemas/layers.schema.json` | `unravel.schema` |
| Batch validation | Runs the pipeline over a directory, reports pass/fail per image | `unravel.validation` |

All model weights are downloaded on first use and cached under `~/.cache/unravel` (override with the `UNRAVEL_CACHE_DIR` environment variable).

## Output contract

For an input illustration, Unravel writes:

- `layers/<label>.png` — one cropped RGBA image per part.
- `layers.json` — part list with semantic label, parent-child hierarchy, depth order (z-index), a canvas-space anchor point, and a canvas-space bounding box per part.

This is the contract shared with the downstream Warp repository; see `schemas/layers.schema.json`. That file documents which invariants (id uniqueness, `parent_id` referential integrity, at-most-one-part-per-label, required labels) are enforced by `unravel.schema.validate_layers_dict` rather than by the JSON Schema itself, since JSON Schema draft-07 cannot express cross-item constraints.

## Installation

```bash
python3 -m pip install -e ".[dev]"
```

Requires Python 3.10+. Dependencies: torch, opencv-python, pillow, numpy, onnxruntime, huggingface_hub, jsonschema.

## Usage

```bash
# Decompose a single illustration
unravel run input.png -o out/

# Also fill occluded regions (e.g. a face patch hidden by hair or an eyebrow) with LaMa inpainting
unravel run input.png -o out/ --inpaint

# Run the pipeline over a directory of test illustrations
unravel validate test_images/ -o out/ --inpaint

# Re-summarize a validation run after manually reviewing out/face_montage.png
# and filling in each image's "artifact_review" (true/false) in validation_report.json
unravel report out/
```

`unravel validate` writes one subdirectory per input image, a `validation_report.json` with per-image pass/fail status, and a `face_montage.png` contact sheet for quick visual review of inpainting quality.

## Known limitations

- Part masks are built from a fixed set of geometric heuristics (ellipse/hull/margin constants in `unravel/parts/masks.py`) tuned against a single fixture image. They may need retuning for illustrations with different proportions or art styles.
- LaMa inpainting can smear the dark eyeliner/lash pixels sitting at an eye mask's boundary into a blotchy patch rather than clean skin, particularly when several small holes (both eyes, both eyebrows, mouth) sit close together in one crop. This is a known, unresolved limitation — see the `refactor: replace diffusion inpainting with LaMa` commit message for what was already tried (larger crop padding, wider mask dilation) before attempting another fix.
- The "no obvious inpainting artifacts" success criterion is not automated; `artifact_review` in `validation_report.json` is a manual field filled in after reviewing `face_montage.png`.

## License

Apache 2.0. See `LICENSE`. Third-party pretrained models keep their own upstream licenses:

- Face detector: [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)
- Landmark regressor: [kanosawa/anime_face_landmark_detection](https://github.com/kanosawa/anime_face_landmark_detection)
- Segmentation: [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
- Inpainting: [advimman/lama](https://github.com/advimman/lama) (via the [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) TorchScript export)
