from pathlib import Path
import time
import cv2
from .config import PipelineConfig
from .decoder import decode, decode_all, decode_qr_regions, decode_with_opencv_barcode
from .variants import generate_variants

def classify_image(path: Path, config: PipelineConfig, expected: str | None = None) -> dict:
    started = time.perf_counter()
    record = {"image_path": str(path.resolve()), "assigned_class": "ERROR", "decoded_value": "",
              "validation_status": "UNVERIFIED", "successful_preprocessing_method": "",
              "processing_time_seconds": 0.0, "errors": ""}
    try:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            record["assigned_class"] = "UNREADABLE"
            record["errors"] = "OpenCV could not read image"
            return record
        found = decode(frame)
        if found:
            found = decode_all(frame)
            record["assigned_class"] = "Good"
            record["decoded_value"] = "; ".join(item["decoded_data"] for item in found)
            record["validation_status"] = _validation(record["decoded_value"], expected)
            return record
        # If the full-frame decode failed, try detected QR regions as a
        # separate preprocessing route before the generic image variants.
        regional = decode_qr_regions(frame)
        if regional:
            record.update(
                assigned_class="Average",
                decoded_value="; ".join(item["decoded_data"] for item in regional),
                validation_status=_validation("; ".join(item["decoded_data"] for item in regional), expected),
                successful_preprocessing_method="; ".join(sorted({item.get("recovery_method", "qr_region_decode") for item in regional})),
            )
            return record
        for attempt, (method, candidate) in enumerate(generate_variants(frame, config)):
            if config.max_attempts is not None and attempt >= config.max_attempts:
                break
            found = decode(candidate)
            decoder_method = "pyzbar"
            if not found:
                found = decode_with_opencv_barcode(candidate)
                decoder_method = "opencv_barcode" if found else decoder_method
            if found:
                record.update(assigned_class="Average", decoded_value=found[0]["decoded_data"],
                              successful_preprocessing_method=f"{method}+{decoder_method}",
                              validation_status=_validation(found[0]["decoded_data"], expected))
                if config.debug:
                    debug_dir = config.output_dir / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(debug_dir / f"{path.stem}_{method}{path.suffix}"), candidate)
                return record
        record["assigned_class"] = "Bad"
        record["validation_status"] = "NO_DECODE"
    except Exception as exc:
        record["assigned_class"] = "ERROR"
        record["errors"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["processing_time_seconds"] = round(time.perf_counter() - started, 6)
    return record

def _validation(decoded: str, expected: str | None) -> str:
    if expected is None or not expected.strip():
        return "UNVERIFIED"
    return "MATCH" if decoded.strip() == expected.strip() else "MISMATCH"
