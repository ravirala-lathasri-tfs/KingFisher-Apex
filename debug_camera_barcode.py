"""Forensic test for the saved camera ROI; does not open a camera."""
from pathlib import Path
import cv2
from barcode.barcode_detector import decode_candidates

ROOT = Path(__file__).parent
path = ROOT / "debug" / "camera_roi.jpg"
image = cv2.imread(str(path))
if image is None:
    raise SystemExit(f"Missing {path}. Run app.py with a barcode visible first.")

height, width = image.shape[:2]
channels = 1 if len(image.shape) == 2 else image.shape[2]
print(f"ROI size: {width} x {height}")
print(f"Channels: {channels}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}, mean: {image.mean():.1f}, std: {image.std():.1f}")
detections, candidates, outcomes, successful = decode_candidates(image)
for name, success in outcomes:
    print(f"Candidate: {name}\nResult: {'SUCCESS' if success else 'FAIL'}")
    if success:
        print(f"Barcode: {detections[0]['barcode']}\nBarcode type: {detections[0]['barcode_type']}")
        cv2.imwrite(str(ROOT / "debug" / "successful_candidate.jpg"), candidates[name])
print(f"\nCAMERA ROI IMAGE: decoder = {'SUCCESS' if detections else 'FAIL'}")
if not detections:
    print("No candidate decoded the saved camera ROI.")
