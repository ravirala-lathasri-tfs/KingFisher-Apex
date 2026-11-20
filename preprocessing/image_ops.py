"""Adapters for existing project operations and small missing image helpers."""
import cv2
import numpy as np
from barcode_preprocessor import (
    adaptive_threshold, denoise_image, enhance_contrast, localize_barcode_candidates,
    order_quad_points, otsu_threshold, rectify_qr, sharpen_for_barcode, upscale_image,
)
from barcode_decoder import detect_1d_barcode_regions

def resize_image(frame: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return upscale_image(frame, scale)

def rotate_image(frame: np.ndarray, angle: float) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def perspective_candidates(frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Reuse existing QR localization/rectification, returning only valid ROIs."""
    results = []
    for index, region in enumerate(localize_barcode_candidates(frame)[:3], 1):
        if len(region.get("corners", [])) != 4:
            continue
        try:
            roi = rectify_qr(frame, region["corners"])
            results.append((f"perspective_roi_{index}", roi))
        except (ValueError, cv2.error):
            continue
    return results

def crop_1d_candidates(frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Decode likely 1D barcode regions with a quiet-zone margin."""
    height, width = frame.shape[:2]
    crops = []
    for index, (x, y, w, h) in enumerate(detect_1d_barcode_regions(frame)[:3], 1):
        pad_x, pad_y = max(5, int(w * 0.10)), max(5, int(h * 0.20))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        if x1 > x0 and y1 > y0:
            crops.append((f"crop_1d_{index}", frame[y0:y1, x0:x1].copy()))
    return crops

def curved_1d_candidates(frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Detect, rectify, and cylindrically unwrap likely curved 1D barcodes.

    OpenCV's BarcodeDetector supplies the barcode quadrilateral. We rectify the
    long edge horizontally, add quiet-zone padding, then use remap to expand
    the compressed sides of a cylindrical label. Each curvature strength is
    generated independently from the rectified crop.
    """
    barcode_module = getattr(cv2, "barcode", None)
    if barcode_module is None or not hasattr(barcode_module, "BarcodeDetector"):
        return []
    try:
        detector = barcode_module.BarcodeDetector()
        if hasattr(detector, "detectMulti"):
            detected, points = detector.detectMulti(frame)
        else:
            detected, points = detector.detect(frame)
    except (cv2.error, AttributeError, TypeError):
        return []
    if not detected or points is None:
        return []

    candidates: list[tuple[str, np.ndarray]] = []
    seen: list[tuple[float, float, float, float]] = []
    for raw_points in np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)[:3]:
        try:
            corners = order_quad_points(raw_points)
        except (ValueError, cv2.error):
            continue
        side_lengths = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
        long_pair = (side_lengths[0] + side_lengths[2])
        short_pair = (side_lengths[1] + side_lengths[3])
        if min(side_lengths) < 8 or min(long_pair, short_pair) <= 0:
            continue
        aspect = max(long_pair, short_pair) / min(long_pair, short_pair)
        if aspect < 1.35 or aspect > 20:
            continue

        x, y, box_w, box_h = cv2.boundingRect(corners.astype(np.int32))
        signature = (float(x), float(y), float(box_w), float(box_h))
        if any(
            abs(x - ox) < 8 and abs(y - oy) < 8
            and abs(box_w - ow) < 12 and abs(box_h - oh) < 12
            for ox, oy, ow, oh in seen
        ):
            continue
        seen.append(signature)

        # Make the barcode's sequence axis horizontal regardless of rotation.
        if short_pair > long_pair:
            corners = np.roll(corners, -1, axis=0)
            side_lengths = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
        target_w = max(240, min(1800, int(round(side_lengths[0] + side_lengths[2]))))
        target_h = max(100, min(1000, int(round(side_lengths[1] + side_lengths[3]))))
        destination = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype=np.float32,
        )
        try:
            matrix = cv2.getPerspectiveTransform(corners, destination)
            rectified = cv2.warpPerspective(
                frame, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
            )
        except cv2.error:
            continue

        pad_x = max(8, int(round(target_w * 0.08)))
        padded = cv2.copyMakeBorder(
            rectified, 0, 0, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        out_h, out_w = padded.shape[:2]
        yy, xx = np.indices((out_h, out_w), dtype=np.float32)
        content_w = float(target_w)
        center = pad_x + content_w / 2.0
        half_width = content_w / 2.0
        normalized = np.clip((xx - center) / max(1.0, half_width), -1.0, 1.0)
        for curvature in (0.55, 0.75, 0.95):
            map_x = center + half_width * np.sin(normalized * curvature) / np.sin(curvature)
            # Preserve the white quiet zone outside the barcode itself.
            map_x = np.where((xx < pad_x) | (xx >= pad_x + target_w), xx, map_x).astype(np.float32)
            unwrapped = cv2.remap(
                padded, map_x, yy, cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
            )
            candidates.append((f"curved_1d_unwrap_{curvature:g}", unwrapped))
    return candidates

def gamma_correct(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Adjust brightness with a lookup table while preserving color channels."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = np.array([min(255, round(((i / 255.0) ** (1.0 / gamma)) * 255)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)

def clahe_otsu(frame: np.ndarray) -> np.ndarray:
    """Apply the existing contrast enhancer, then the existing Otsu threshold."""
    return otsu_threshold(enhance_contrast(frame))
