"""Single-image verification for barcode preprocessing and decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from barcode_decoder import decode_barcodes
from barcode_preprocessor import (
    analyze_image_quality,
    describe_adaptive_candidates,
    generate_roi_candidates,
    localize_barcode_candidates,
    localize_qr_roi,
)


def annotate(frame: Any, detections: list[dict[str, Any]]) -> Any:
    """Draw results when the decoder module exposes its optional helper."""
    try:
        from barcode_decoder import draw_barcodes
        return draw_barcodes(frame, detections)
    except (ImportError, AttributeError, KeyError, TypeError):
        result = frame.copy()
        for detection in detections:
            cv2.putText(
                result,
                f"{detection.get('barcode_type', 'UNKNOWN')}: {detection.get('decoded_data', '')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        return result


def print_detections(detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        print(f"Type: {detection.get('barcode_type', 'UNKNOWN')}")
        print(f"Value: {detection.get('decoded_data', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test one barcode image")
    parser.add_argument("image", help="Path to one input image")
    args = parser.parse_args()
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 1

    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    quality = analyze_image_quality(frame)
    print(f"Image path: {image_path}")
    print(f"Image size: {width} x {height}")
    print(f"Channels: {channels}")
    print("\nQuality analysis:")
    for key in ("sharpness", "contrast", "brightness", "dynamic_range", "noise", "edge_strength"):
        if key in quality:
            print(f"{key.replace('_', ' ').title()}: {quality[key]:.3f}")
    print(f"Status: {quality.get('quality_status', 'UNKNOWN')}")
    print(f"Problems: {', '.join(quality.get('problems', [])) or 'None'}")

    print("\nTrying original image...")
    detections = decode_barcodes(frame.copy())
    successful_frame = frame.copy()
    used = "NONE"
    preprocessing_required = "NO"
    if detections:
        print("SUCCESS")
        print("\nDetected barcodes:")
        print_detections(detections)
    else:
        print("FAILED")
        print("\nGenerating adaptive preprocessing candidates...")
        candidates = describe_adaptive_candidates(frame, include_decoder_fallback=True)
        print(f"Candidates generated: {len(candidates)}")
        report_problems = quality.get("problems", [])
        if report_problems:
            print("\nPreprocessing decision:")
            print(f"Primary issue: {report_problems[0]}")
            print(f"Recommended operation: {'UPSCALE' if report_problems[0] == 'SMALL_BARCODE' else report_problems[0]}")
            print(f"Reason: diagnosed problems are {', '.join(report_problems)}")
        for index, (candidate_name, candidate) in enumerate(candidates[1:], start=1):
            print(f"\nCandidate {index}: {candidate_name}")
            print(f"  Size: {candidate.shape[1]} x {candidate.shape[0]}")
            candidate_detections = decode_barcodes(candidate.copy())
            if candidate_detections:
                print("SUCCESS")
                detections = candidate_detections
                successful_frame = candidate.copy()
                used = candidate_name
                preprocessing_required = "YES"
                print("\nDetected barcodes:")
                print_detections(detections)
                break
            print("FAILED")

    if not detections:
        localizations = localize_barcode_candidates(frame)
        for localization in localizations:
            x, y, width, height = localization["bbox"]
            print(f"\nBarcode localization: {localization['type']}")
            print(f"ROI: x={x}, y={y}, w={width}, h={height}")
            roi = frame[max(0, y):y + height, max(0, x):x + width].copy()
            output_dir = Path("test_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "detected_roi.jpg"), roi)
            roi_candidates = generate_roi_candidates(frame, localization["corners"])
            for candidate_name, candidate in roi_candidates:
                print(f"\nTrying ROI candidate: {candidate_name}")
                candidate_detections = decode_barcodes(candidate.copy())
                if candidate_detections:
                    print("SUCCESS")
                    detections = candidate_detections
                    successful_frame = candidate.copy()
                    used = candidate_name
                    preprocessing_required = "YES"
                    print("\nDetected barcodes:")
                    print_detections(detections)
                    break
                print("FAILED")
            if detections:
                break

    output_dir = Path("test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_output = output_dir / f"{image_path.stem}_original{image_path.suffix}"
    result_output = output_dir / f"{image_path.stem}_result{image_path.suffix}"
    cv2.imwrite(str(original_output), frame)
    cv2.imwrite(str(result_output), annotate(successful_frame, detections))
    try:
        cv2.imshow("Barcode test result", annotate(successful_frame, detections))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Note: image display is unavailable in this environment; files were saved.")

    print("\n================ TEST RESULT ================")
    print(f"Original image decoded: {'YES' if used == 'NONE' and detections else 'NO'}")
    print(f"Preprocessing required: {preprocessing_required}")
    print(f"Preprocessing used: {used}")
    print(f"Barcode(s) detected: {len(detections)}")
    print("Test status: PASS" if detections else "Test status: NO BARCODE DECODED")
    print(f"Saved original: {original_output}")
    print(f"Saved result: {result_output}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
