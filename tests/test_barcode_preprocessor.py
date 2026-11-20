import numpy as np

from barcode_preprocessor import (
    analyze_image_quality,
    apply_adaptive_preprocessing,
    describe_adaptive_candidates,
    generate_adaptive_candidates,
)


def quality(width, height):
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    return analyze_image_quality(frame)


def barcode_like_frame(width, height):
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_width_159_is_small():
    assert "SMALL_BARCODE" in quality(159, 300)["problems"]


def test_width_160_is_not_small():
    assert "SMALL_BARCODE" not in quality(160, 300)["problems"]


def test_width_161_is_not_small():
    assert "SMALL_BARCODE" not in quality(161, 300)["problems"]


def test_short_height_does_not_trigger_small_when_width_is_large():
    assert "SMALL_BARCODE" not in quality(300, 100)["problems"]


def test_large_height_does_not_hide_small_width():
    assert "SMALL_BARCODE" in quality(100, 300)["problems"]


def test_width_100_upscales_to_minimum_width_and_preserves_aspect_ratio():
    frame = barcode_like_frame(100, 170)
    candidates = generate_adaptive_candidates(frame)
    upscaled = candidates[1]
    assert upscaled.shape[1] >= 160
    assert abs((upscaled.shape[1] / upscaled.shape[0]) - (100 / 170)) < 0.01


def test_width_80_uses_required_scale_of_2():
    frame = barcode_like_frame(80, 200)
    upscaled = generate_adaptive_candidates(frame)[1]
    assert upscaled.shape[1] == 160
    assert upscaled.shape[0] == 400


def test_width_159_upscales_to_at_least_minimum_width():
    frame = barcode_like_frame(159, 200)
    assert generate_adaptive_candidates(frame)[1].shape[1] >= 160


def test_width_160_does_not_upscale():
    frame = barcode_like_frame(160, 100)
    assert all(candidate.shape[:2] == frame.shape[:2] for candidate in generate_adaptive_candidates(frame))


def test_width_over_160_does_not_upscale():
    frame = barcode_like_frame(200, 100)
    assert all(candidate.shape[:2] == frame.shape[:2] for candidate in generate_adaptive_candidates(frame))


def low_contrast_frame():
    frame = np.full((180, 520, 3), 110, dtype=np.uint8)
    frame[:, ::2] = 130
    return frame


def test_low_contrast_generates_exact_ordered_candidates():
    frame = low_contrast_frame()
    report = analyze_image_quality(frame)
    assert "LOW_CONTRAST" in report["problems"]
    labels = [label for label, _ in describe_adaptive_candidates(frame)]
    assert labels == [
        "ORIGINAL",
        "CLAHE",
        "CLAHE + OTSU",
        "CLAHE + UPSCALE 2X",
        "CLAHE + UPSCALE 2X + OTSU",
    ]


def test_decoder_selection_stops_at_first_successful_candidate():
    frame = low_contrast_frame()
    attempted = []

    def selector(candidate):
        attempted.append(candidate.shape[:2])
        return len(attempted) == 3

    selected, report = apply_adaptive_preprocessing(frame, selector=selector)
    assert selected is not None
    assert len(attempted) == 3
    assert report["selected_candidate"] == "CLAHE + OTSU"
    assert report["preprocessing_applied"] == ["CLAHE + OTSU"]


def test_original_success_does_not_attempt_fallbacks():
    frame = low_contrast_frame()
    attempts = []
    selected, report = apply_adaptive_preprocessing(frame, selector=lambda candidate: not attempts.append(candidate))
    assert selected is not None
    assert len(attempts) == 1
    assert report["selected_candidate"] == "ORIGINAL"


def test_all_decoder_candidates_can_fail_without_selecting_processing():
    frame = low_contrast_frame()
    selected, report = apply_adaptive_preprocessing(frame, selector=lambda _: False)
    assert selected is not None
    assert report["selected_candidate"] == "ORIGINAL"
    assert report["preprocessing_applied"] == []


def test_blurry_generates_ordered_sharpen_fallbacks():
    frame = np.full((190, 560, 3), 128, dtype=np.uint8)
    for start in range(0, 560, 80):
        frame[:, start:start + 40] = 30
        frame[:, start + 40:start + 80] = 220
    frame = __import__("cv2").GaussianBlur(frame, (11, 11), 0)
    labels = [label for label, _ in describe_adaptive_candidates(frame)]
    assert labels == [
        "ORIGINAL",
        "SHARPEN (mild)",
        "SHARPEN + UPSCALE 2X",
        "SHARPEN + UPSCALE 2X + OTSU",
    ]
