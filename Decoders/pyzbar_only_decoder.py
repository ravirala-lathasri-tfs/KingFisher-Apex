"""Decode barcodes with Pyzbar only.

This module intentionally performs no preprocessing, resizing, enhancement,
cropping, or recovery.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
from pyzbar.pyzbar import decode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def decode_image(image_path: str | Path) -> list[dict[str, Any]]:
    """Decode all Pyzbar-supported barcodes in one original image."""
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return []

    results: list[dict[str, Any]] = []
    for item in decode(image):
        results.append(
            {
                "decoded_data": item.data.decode("utf-8", errors="replace"),
                "barcode_type": str(item.type),
                "bbox": (item.rect.left, item.rect.top, item.rect.width, item.rect.height),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode images using Pyzbar only.")
    parser.add_argument("path", type=Path, help="Image file or directory to test")
    args = parser.parse_args()

    if args.path.is_file():
        files = [args.path]
    elif args.path.is_dir():
        files = sorted(
            path for path in args.path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        parser.error(f"Path does not exist: {args.path}")

    decoded_count = 0
    for path in files:
        results = decode_image(path)
        if results:
            decoded_count += 1
            print(f"DECODED: {path}")
            for result in results:
                print(f"  Type: {result['barcode_type']}")
                print(f"  Value: {result['decoded_data']}")
        else:
            print(f"NOT_DECODED: {path}")

    print("\n========== SUMMARY ==========")
    print(f"Images tested: {len(files)}")
    print(f"Decoded: {decoded_count}")
    print(f"Not decoded: {len(files) - decoded_count}")


if __name__ == "__main__":
    main()
