from unravel.parts import DEPTH_ORDER

FACE_OVERLAY_LABELS = ("eyebrow_l", "eyebrow_r", "eye_l", "eye_r", "mouth")


def test_all_labels_ranked():
    expected_labels = {"face", "eye_l", "eye_r", "eyebrow_l", "eyebrow_r", "mouth", "hair_front", "hair_back"}
    assert set(DEPTH_ORDER) == expected_labels


def test_hair_back_is_bottommost():
    assert DEPTH_ORDER["hair_back"] < DEPTH_ORDER["face"]


def test_hair_front_is_topmost():
    for label in ("face", *FACE_OVERLAY_LABELS, "hair_back"):
        assert DEPTH_ORDER["hair_front"] > DEPTH_ORDER[label]


def test_face_overlays_sit_above_face():
    for label in FACE_OVERLAY_LABELS:
        assert DEPTH_ORDER[label] > DEPTH_ORDER["face"]
