"""Reusable 1D/2D barcode decoder based on OpenCV and Pyzbar/ZBar."""

from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path

import cv2
import numpy as np
from barcode_preprocessor import order_quad_points, rectify_qr
from barcode_scanner.runtime import configure_barcode_runtime

configure_barcode_runtime()
from pyzbar.pyzbar import decode as pyzbar_decode


def _decode_bytes(data: bytes) -> str:
    """Convert barcode bytes to text without failing on invalid UTF-8."""
    return data.decode("utf-8", errors="replace")


def decode_barcodes(frame: np.ndarray | None) -> list[dict[str, Any]]:
    """Decode every 1D/2D barcode in an OpenCV image.

    The decoder is intentionally format-agnostic: no ZBar symbol filter is
    supplied, so all formats supported by the installed Pyzbar/ZBar build are
    considered. Duplicate results are removed using type, text, and location.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    try:
        decoded = pyzbar_decode(frame)
    except Exception:
        # Decoder failures should not crash a caller's frame-processing loop.
        return []

    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in decoded:
        text = _decode_bytes(item.data)
        key = (str(item.type), text)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "decoded_data": text,
                "barcode_type": str(item.type),
                "bbox": (int(item.rect.left), int(item.rect.top), int(item.rect.width), int(item.rect.height)),
                "polygon": [(int(point.x), int(point.y)) for point in (item.polygon or [])],
            }
        )
    return results


def detect_1d_barcode_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find likely parallel-bar regions without attempting to decode them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    magnitude = cv2.convertScaleAbs(np.abs(gradient))
    _, mask = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[tuple[int, int, int, int]] = []
    image_area = gray.shape[0] * gray.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.001:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        ratio = max(width, height) / max(1, min(width, height))
        if ratio < 1.5 or width < 20 or height < 20:
            continue
        regions.append((x, y, width, height))
    return _deduplicate_regions(regions)


def _deduplicate_regions(regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for region in sorted(regions, key=lambda item: item[2] * item[3], reverse=True):
        x, y, w, h = region
        duplicate = False
        for ox, oy, ow, oh in kept:
            ix, iy = max(x, ox), max(y, oy)
            ax, ay = min(x + w, ox + ow), min(y + h, oy + oh)
            intersection = max(0, ax - ix) * max(0, ay - iy)
            union = w * h + ow * oh - intersection
            if union and intersection / union > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def _matches_detection(region: tuple[int, int, int, int], detection: dict[str, Any]) -> bool:
    bbox = detection.get("bbox")
    if not bbox:
        return False
    x, y, w, h = region
    ox, oy, ow, oh = bbox
    ix, iy = max(x, ox), max(y, oy)
    ax, ay = min(x + w, ox + ow), min(y + h, oy + oh)
    intersection = max(0, ax - ix) * max(0, ay - iy)
    return intersection / max(1, min(w * h, ow * oh)) > 0.15


def _same_physical_detection(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = left.get("bbox"), right.get("bbox")
    if not a or not b:
        return left.get("barcode_type") == right.get("barcode_type") and left.get("decoded_data") == right.get("decoded_data")
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - intersection
    contained = intersection / max(1, min(aw * ah, bw * bh))
    return union > 0 and (intersection / union > 0.35 or contained > 0.7)


def merge_unique_physical_detections(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge detections by physical overlap, retaining distinct equal values."""
    merged: list[dict[str, Any]] = []
    for group in groups:
        for detection in group:
            if not any(_same_physical_detection(detection, existing) for existing in merged):
                merged.append(detection)
    return merged


def detect_multiple_qr_regions(frame: np.ndarray) -> list[dict[str, Any]]:
    """Return all QR quadrilaterals OpenCV can localize, without decoding them."""
    detector = cv2.QRCodeDetector()
    regions: list[dict[str, Any]] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    variants: list[tuple[str, np.ndarray, float]] = [("ORIGINAL", frame, 1.0), ("GRAY", gray, 1.0)]
    if min(gray.shape[:2]) > 900:
        for scale in (0.5, 0.75, 1.5, 2.0):
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
            variants.append((f"GRAY_{scale:g}X", small, scale))
            if scale <= 1.0:
                variants.append((f"BLUR_{scale:g}X", cv2.GaussianBlur(small, (5, 5), 0), scale))
    for method, variant, scale in variants:
      try:
        result = detector.detectAndDecodeMulti(variant)
        if len(result) == 4:
            found, _, points, _ = result
        else:
            found, points = False, None
        if found and points is not None:
            for item in np.asarray(points, dtype=np.float32):
                corners = item.reshape(4, 2) / scale
                x, y, width, height = cv2.boundingRect(corners.reshape(-1, 1, 2))
                regions.append({"corners": corners, "bbox": (x, y, width, height), "method": f"QR_MULTI_{method}"})
      except (AttributeError, cv2.error, ValueError):
        continue
    if not regions:
        # Localization-only fallback for dense screen/moire structure.
        smooth = cv2.GaussianBlur(gray, (7, 7), 0)
        dark = cv2.threshold(smooth, 95, 255, cv2.THRESH_BINARY_INV)[1]
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = gray.shape[0] * gray.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < image_area * 0.02:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < min(gray.shape[:2]) * 0.15:
                continue
            corners = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)
            regions.append({"corners": corners, "bbox": (x, y, width, height), "method": "DARK_STRUCTURE"})
            break
    try:
        found, points = detector.detectMulti(frame)
        if found and points is not None:
            for item in np.asarray(points, dtype=np.float32):
                corners = item.reshape(4, 2)
                x, y, width, height = cv2.boundingRect(corners.reshape(-1, 1, 2))
                if not any(_matches_detection((x, y, width, height), {"bbox": old["bbox"]}) for old in regions):
                    regions.append({"corners": corners, "bbox": (x, y, width, height), "method": "QR_DETECT_MULTI"})
    except (AttributeError, cv2.error, ValueError):
        pass
    unique: list[dict[str, Any]] = []
    for region in regions:
        if not any(_matches_detection(region["bbox"], {"bbox": old["bbox"]}) and _matches_detection(old["bbox"], {"bbox": region["bbox"]}) for old in unique):
            unique.append(region)
    return unique


def decode_multiple_qr_regions(frame: np.ndarray | None) -> dict[str, Any]:
    """Decode independently localized QR regions and retain failed regions."""
    original = decode_barcodes(frame)
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return {"results": original, "regions": [], "failed_regions": []}
    regions = detect_multiple_qr_regions(frame)
    decoded = list(original)
    failed_regions: list[dict[str, Any]] = []
    debug_root = Path("test_output")
    candidates_dir = debug_root / "qr_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    localization_debug = frame.copy()
    debug_records: list[dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        candidate_id = f"QR-{index:03d}"
        region["id"] = candidate_id
        corners = np.asarray(region["corners"], dtype=np.float32).reshape(4, 2)
        x, y, width, height = region["bbox"]
        cv2.polylines(localization_debug, [corners.astype(np.int32)], True, (0, 255, 255), 3)
        cv2.putText(localization_debug, candidate_id, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        roi = frame[max(0, y):min(frame.shape[0], y + height), max(0, x):min(frame.shape[1], x + width)].copy()
        cv2.imwrite(str(candidates_dir / f"{candidate_id}_original.png"), roi)
        ordered = order_quad_points(corners)
        side_lengths = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
        record = {"id": candidate_id, "method": region.get("method"), "bbox": region["bbox"], "corners": ordered.tolist(), "side_lengths": side_lengths}
        if any(_matches_detection(region["bbox"], item) for item in original):
            region["status"] = "DECODED_ORIGINAL"
            record["status"] = "ALREADY_DECODED"
            debug_records.append(record)
            continue
        output_size = max(300, min(1000, int(max(width, height) * 2.0)))
        try:
            roi = rectify_qr(frame, corners, output_size=output_size, padding_ratio=0.05)
            cv2.imwrite(str(candidates_dir / f"{candidate_id}_rectified.png"), roi)
        except (ValueError, cv2.error):
            record["status"] = "INVALID_RECTIFICATION"
            debug_records.append(record)
            failed_regions.append(region)
            continue
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, otsu_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        variants = [("original", roi), ("grayscale", gray_roi), ("otsu", otsu_roi), ("adaptive", adaptive_roi)]
        outcomes = {}
        found = []
        for name, variant in variants:
            result = decode_barcodes(variant)
            outcomes[name] = bool(result)
            if result and not found:
                found = result
        try:
            opencv_text, _, _ = cv2.QRCodeDetector().detectAndDecode(roi)
        except cv2.error:
            opencv_text = ""
        record.update({"status": "RECOVERED" if found else "FAILED", "decode_outcomes": outcomes, "opencv_success": bool(opencv_text), "opencv_text": opencv_text, "output_size": output_size})
        debug_records.append(record)
        if found:
            region["status"] = "RECOVERED"
            for item in found:
                item["recovered"] = True
                item["region_bbox"] = region["bbox"]
                if not any(_matches_detection(region["bbox"], old) for old in decoded):
                    decoded.append(item)
        else:
            region["status"] = "FAILED"
            failed_regions.append(region)
    cv2.imwrite(str(debug_root / "qr_localization_debug.jpg"), localization_debug)
    return {"results": decoded, "regions": regions, "failed_regions": failed_regions, "debug_records": debug_records}


def decode_barcodes_with_recovery(frame: np.ndarray | None) -> dict[str, Any]:
    """Decode originals first, then recover only unmatched barcode ROIs."""
    original = decode_barcodes(frame)
    if frame is None or not original:
        original = original or []
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return {"results": original, "original_results": original, "recovered_results": [], "regions": []}

    recovered: list[dict[str, Any]] = []
    regions = detect_1d_barcode_regions(frame)
    candidate_records: list[dict[str, Any]] = []
    for region in regions:
        matched = any(_matches_detection(region, item) for item in original)
        record = {"localized": True, "region": region, "matched_global": matched, "status": "ALREADY_DECODED" if matched else "UNDECODED_CANDIDATE"}
        candidate_records.append(record)
        if matched:
            continue
        x, y, w, h = region
        pad_x, pad_y = max(4, int(w * 0.08)), max(4, int(h * 0.12))
        crop = frame[max(0, y - pad_y):min(frame.shape[0], y + h + pad_y), max(0, x - pad_x):min(frame.shape[1], x + w + pad_x)]
        crop_x, crop_y = max(0, x - pad_x), max(0, y - pad_y)
        for candidate, scale in [(crop, 1.0), (cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0)]:
            found = decode_barcodes(candidate)
            if found:
                record["status"] = "RECOVERED"
                for item in found:
                    item["recovered"] = True
                    if item.get("bbox"):
                        bx, by, bw, bh = item["bbox"]
                        item["bbox"] = (int(crop_x + bx / scale), int(crop_y + by / scale), int(bw / scale), int(bh / scale))
                    recovered.append(item)
                break
        else:
            record["status"] = "FAILED"
    recovered = merge_unique_physical_detections(recovered)
    merged = merge_unique_physical_detections(original, recovered)
    return {
        "results": merged,
        "original_results": original,
        "recovered_results": recovered,
        "regions": regions,
        "candidate_records": candidate_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode 1D/2D barcodes from an image")
    parser.add_argument("image", help="Path to an image file")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 1

    report = decode_barcodes_with_recovery(frame)
    qr_report = decode_multiple_qr_regions(frame)
    if not report["results"] and qr_report["results"]:
        report["results"] = qr_report["results"]
        report["recovered_results"] = [item for item in qr_report["results"] if item.get("recovered")]
    report["qr_regions"] = qr_report["regions"]
    report["qr_failed_regions"] = qr_report["failed_regions"]
    report["qr_debug_records"] = qr_report.get("debug_records", [])
    detections = report["results"]
    print(f"Original detections: {len(report['original_results'])}")
    print(f"Recovered detections: {len(report['recovered_results'])}")
    records = report.get("candidate_records", [])
    qr_records = report.get("qr_debug_records", [])
    localized_count = len(records) if records else len(qr_records)
    already = sum(r["status"] == "ALREADY_DECODED" for r in records)
    attempts = sum(r["status"] != "ALREADY_DECODED" for r in records)
    failed = sum(r["status"] == "FAILED" for r in records)
    print("\n========================================")
    print("RESULT ACCOUNTING")
    print("========================================")
    print(f"Original detections:        {len(report['original_results'])}")
    print(f"Localized physical ROIs:    {localized_count}")
    print(f"Already decoded ROIs:       {already}")
    print(f"New ROI decode attempts:    {attempts}")
    print(f"Recovered detections:       {len(report['recovered_results'])}")
    print(f"Failed ROI decodes:         {failed}")
    print(f"Final unique detections:    {len(report['results'])}")
    print(f"QR localization candidates: {len(report.get('qr_regions', []))}")
    print(f"Failed QR ROI candidates:    {len(report.get('qr_failed_regions', []))}")
    for record in report.get("qr_debug_records", []):
        print(f"{record['id']}: method={record['method']}, status={record['status']}, corners={record['corners']}")
        if "decode_outcomes" in record:
            print(f"  pyzbar={record['decode_outcomes']}, opencv={'SUCCESS' if record['opencv_success'] else 'FAILED'}")
    print(f"Original + recovered:       {len(report['original_results']) + len(report['recovered_results'])}")
    if len(report["results"]) > len(report["original_results"]) + len(report["recovered_results"]):
        print("WARNING: RESULT ACCOUNTING MISMATCH")
    for index, record in enumerate(records, start=1):
        print(f"ROI-{index:03d}: matches original={'YES' if record['matched_global'] else 'NO'}, status={record['status']}")
    if not detections:
        print("No barcodes detected.")
    for detection in detections:
        print(f"Barcode type: {detection['barcode_type']}")
        print(f"Decoded value: {detection['decoded_data']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
