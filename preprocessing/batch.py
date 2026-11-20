"""Batch classification, original-preserving class copies, and CSV reporting."""
import argparse
import csv
import logging
import os
import shutil
from pathlib import Path
from .config import IMAGE_EXTENSIONS, PipelineConfig
from .pipeline import classify_image

LOG = logging.getLogger("preprocessing")
FIELDS = ["image_path", "assigned_class", "decoded_value", "validation_status",
          "successful_preprocessing_method", "processing_time_seconds", "errors"]

def load_expected(csv_path: Path | None) -> dict[str, str]:
    if not csv_path: return {}
    values = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image = row.get("image_path") or row.get("filename") or row.get("image")
            value = row.get("expected_value") or row.get("expected") or row.get("barcode")
            if image and value: values[Path(image).name] = value
    return values

def run_batch(inputs: list[Path], config: PipelineConfig, expected: dict[str, str] | None = None) -> list[dict]:
    expected = expected or {}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    common_root = Path(os.path.commonpath([str(path.resolve().parent) for path in inputs])) if inputs else Path.cwd()
    for source in inputs:
        record = classify_image(source, config, expected.get(source.name))
        report.append(record)
        bucket = record["assigned_class"]
        try:
            relative = source.resolve().relative_to(common_root)
        except ValueError:
            relative = Path(source.name)
        destination = config.output_dir / bucket / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            record["errors"] = (record["errors"] + "; " if record["errors"] else "") + f"copy: {exc}"
        LOG.info("%s -> %s", source, bucket)
    with (config.output_dir / "report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(report)
    return report

def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def main() -> int:
    parser = argparse.ArgumentParser(description="Classify barcode images with Pyzbar and preprocessing")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bad-dir", type=Path, help="Process only the current Bad image folder")
    mode.add_argument("--dataset", type=Path, help="Full dataset root with good/ and bad/ folders")
    parser.add_argument("--output", type=Path, default=Path("preprocessing_output"))
    parser.add_argument("--expected-csv", type=Path)
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum variants per image (default: all)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = PipelineConfig(output_dir=args.output, max_attempts=max(1, args.max_attempts) if args.max_attempts is not None else None, debug=args.debug, expected_csv=args.expected_csv)
    inputs = collect_images(args.bad_dir) if args.bad_dir else collect_images(args.dataset)
    # Full mode reevaluates every image so class assignment follows decode results.
    expected = load_expected(args.expected_csv)
    report = run_batch(inputs, config, expected)
    LOG.info("Processed %d images. Results: %s", len(report), config.output_dir.resolve())
    return 0

if __name__ == "__main__": raise SystemExit(main())
