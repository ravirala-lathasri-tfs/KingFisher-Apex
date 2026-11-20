"""Pyzbar adapters using the repository's established decoder implementation."""
import cv2
import numpy as np
from barcode_decoder import decode_barcodes, detect_multiple_qr_regions
from barcode_preprocessor import adaptive_threshold, enhance_contrast, otsu_threshold, rectify_qr

def decode(frame):
    return decode_barcodes(frame)

def decode_with_opencv_barcode(frame):
    """Decode 1D symbols with OpenCV's barcode engine as a fallback."""
    barcode_module = getattr(cv2, "barcode", None)
    if frame is None or not hasattr(frame, "size") or frame.size == 0 or barcode_module is None:
        return []
    try:
        detector = barcode_module.BarcodeDetector()
        decoded = detector.detectAndDecodeWithType(frame)
        if len(decoded) < 3 or not decoded[0]:
            return []
        values, types = decoded[1], decoded[2]
        points = decoded[3] if len(decoded) > 3 else None
    except (cv2.error, AttributeError, TypeError, ValueError):
        return []

    results = []
    values = values if isinstance(values, (tuple, list)) else (values,)
    types = types if isinstance(types, (tuple, list)) else (types,)
    quads = [] if points is None else list(np.asarray(points).reshape(-1, 4, 2))
    for index, value in enumerate(values):
        if not value:
            continue
        kind = types[index] if index < len(types) and types[index] else "BARCODE"
        bbox = (0, 0, int(frame.shape[1]), int(frame.shape[0]))
        if index < len(quads):
            x, y, w, h = cv2.boundingRect(quads[index].astype(np.float32))
            bbox = (int(x), int(y), int(w), int(h))
        results.append({"decoded_data": str(value), "barcode_type": str(kind), "bbox": bbox, "polygon": []})
    return results

def decode_qr_regions(frame):
    """Decode QR codes independently so one successful code doesn't hide others."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []
    results = []
    for region in detect_multiple_qr_regions(frame):
        x, y, width, height = region["bbox"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        found = decode_barcodes(crop)
        for item in found:
            bx, by, bw, bh = item["bbox"]
            item["bbox"] = (x0 + bx, y0 + by, bw, bh)
            item["polygon"] = [(x0 + px, y0 + py) for px, py in item.get("polygon", [])]
            item["recovery_method"] = "qr_region_crop"
        if not found:
            try:
                rectified = rectify_qr(frame, region["corners"], output_size=900, padding_ratio=0.05)
            except (ValueError, cv2.error):
                continue
            gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
            candidates = (
                ("qr_rectified_grayscale", gray),
                ("qr_rectified_clahe", enhance_contrast(gray)),
                ("qr_rectified_otsu", otsu_threshold(gray)),
                ("qr_rectified_adaptive", adaptive_threshold(gray)),
                ("qr_rectified_clahe_otsu", otsu_threshold(enhance_contrast(gray))),
            )
            for method, candidate in candidates:
                found = decode_barcodes(candidate)
                if found:
                    for item in found:
                        item["bbox"] = region["bbox"]
                        item["recovery_method"] = method
                    break
        results.extend(found)
    return results

def decode_all(frame):
    """Return full-frame decodes plus additional QR region results, deduplicated."""
    results = decode_barcodes(frame)
    if not any(item.get("barcode_type") == "QRCODE" for item in results):
        return results
    seen = {(item.get("barcode_type"), item.get("decoded_data")) for item in results}
    for item in decode_qr_regions(frame):
        key = (item.get("barcode_type"), item.get("decoded_data"))
        if key not in seen:
            results.append(item)
            seen.add(key)
    return results
