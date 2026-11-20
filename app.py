import json, logging, time
from pathlib import Path
import cv2
from camera.camera_manager import CameraManager
from barcode.barcode_detector import decode_candidates
from sample_data.json_sample_provider import JsonSampleProvider
from processing.scan_manager import ScanManager
from storage.result_store import ResultStore

ROOT = Path(__file__).parent

def get_scan_region(frame, roi_width=900, roi_height=300):
    height, width = frame.shape[:2]
    roi_width = min(int(roi_width), width)
    roi_height = min(int(roi_height), height)
    x1, y1 = (width - roi_width) // 2, (height - roi_height) // 2
    return x1, y1, x1 + roi_width, y1 + roi_height

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (ROOT / "config/config.json").open(encoding="utf-8") as f: config = json.load(f)
    provider = JsonSampleProvider(ROOT / config["sample_database"])
    manager = ScanManager(provider, config["expected_sample_type"])
    store = ResultStore(ROOT / config["output_csv"])
    camera = CameraManager(config.get("camera_index", 0))
    last_barcode = "None"
    # Keep only the main camera window visible by default. Press D to show
    # the optional cropped ROI debug window.
    debug_roi = False
    previous_time = time.perf_counter()
    fps = 0.0
    stable_value, stable_count = None, 0
    debug_index, last_debug_save = 0, 0.0
    debug_names = ["Original", "Grayscale", "Upscaled", "CLAHE", "Sharpened", "Otsu", "Adaptive", "Inverted"]
    try:
        camera.open(); logging.info("Application started; camera initialized")
        while True:
            frame = camera.read()
            if frame is None: logging.warning("Camera frame failure"); break
            x1, y1, x2, y2 = get_scan_region(frame, config.get("roi_width", 900), config.get("roi_height", 300))
            roi = frame[y1:y2, x1:x2]
            detections, candidate_images, outcomes, successful_candidate = decode_candidates(roi)
            if not detections:
                now_debug = time.time()
                if now_debug - last_debug_save >= 3:
                    debug_dir = ROOT / "debug"; debug_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(debug_dir / "camera_raw.jpg"), frame)
                    cv2.imwrite(str(debug_dir / "camera_roi.jpg"), roi)
                    cv2.imwrite(str(debug_dir / "camera_gray.jpg"), candidate_images["Grayscale"])
                    cv2.imwrite(str(debug_dir / "camera_upscaled.jpg"), candidate_images["Upscaled"])
                    cv2.imwrite(str(debug_dir / "camera_threshold.jpg"), candidate_images["Otsu"])
                    cv2.imwrite(str(debug_dir / "camera_failed_barcode.jpg"), roi)
                    last_debug_save = now_debug
                h, w = roi.shape[:2]
                logging.info("Camera ROI: width=%s height=%s channels=%s dtype=%s min=%s max=%s mean=%.1f; Decode result: FAILED; candidates=%s",
                             w, h, 1 if len(roi.shape) == 2 else roi.shape[2], roi.dtype, roi.min(), roi.max(), roi.mean(), outcomes)
            else:
                logging.info("GOOD: barcode decoded using %s: %s", successful_candidate,
                             [(d["barcode"], d["barcode_category"]) for d in detections])
            if detections:
                current = detections[0]["barcode"]
                stable_count = stable_count + 1 if current == stable_value else 1
                stable_value = current
            else:
                stable_value, stable_count = None, 0
            if detections:
                status = f"GOOD: {detections[0]['barcode_category']}"
                status_color = (0, 220, 0)
            else:
                status = "NO BARCODE DETECTED"
                status_color = (0, 0, 255)
            for detection in detections:
                x, y, w, h = detection["bbox"]
                x += x1; y += y1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, detection["barcode"], (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
                last_barcode = detection["barcode"]
                if stable_count < 3:
                    continue
                result = manager.process(detection["barcode"], detection["barcode_type"])
                if result: store.save(result); print(result)
            # Centered guide showing where to place the barcode in the live feed.
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(frame, "PLACE BARCODE HERE", (x1 + 10, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 180, 0), 2)
            now = time.perf_counter()
            elapsed = now - previous_time
            if elapsed > 0: fps = .9 * fps + .1 / elapsed
            previous_time = now
            # Readable status panel: this makes it obvious whether the barcode
            # is inside the camera view and currently decodable.
            cv2.rectangle(frame, (0, 0), (430, 105), (25, 25, 25), -1)
            cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, status_color, 2)
            cv2.putText(frame, f"Current: {last_barcode}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            cv2.putText(frame, f"Camera: Connected  Resolution: {width}x{height}  FPS: {fps:.1f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, .5, (220,220,220), 1)
            cv2.putText(frame, f"ROI: ({x1},{y1})-({x2},{y2})  Processed: {manager.processed_count}", (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (220,220,220), 1)
            cv2.imshow("Barcode Scanner - Camera", frame)
            if debug_roi:
                cv2.imshow("Barcode Processing ROI", candidate_images[debug_names[debug_index]])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r"):
                manager.reset_session()
                camera.release()
                camera.open()
                logging.info("Camera reinitialized")
            if key == ord("d"):
                if debug_roi: debug_index = (debug_index + 1) % len(debug_names)
                else: debug_roi = True
                if not debug_roi: cv2.destroyWindow("Barcode Processing ROI")
    finally:
        camera.release(); cv2.destroyAllWindows(); logging.info("Application stopped")

if __name__ == "__main__": main()
