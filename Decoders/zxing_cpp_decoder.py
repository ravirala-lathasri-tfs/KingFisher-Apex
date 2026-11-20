"""Decode barcode images directly with the ZXing-C++ Python binding.

No preprocessing is performed. The only image transform is none; ZXing-C++
receives the original image pixels loaded by Pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import zxingcpp
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def decode_image(image_path: Path) -> list[dict[str, str]]:
    """Decode all barcodes in one unchanged image using ZXing-C++."""
    with Image.open(image_path) as image:
        barcodes = zxingcpp.read_barcodes(image)

    return [
        {
            "value": barcode.text,
            "format": str(barcode.format),
            "content_type": str(barcode.content_type),
        }
        for barcode in barcodes
    ]


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode one image or all images in a folder with ZXing-C++ only."
    )
    parser.add_argument("input", type=Path, help="Image file or folder containing images")
    args = parser.parse_args()

    images = collect_images(args.input)
    if not images:
        parser.error(f"No image files found at: {args.input}")

    total = 0
    for image_path in images:
        try:
            results = decode_image(image_path)
        except Exception as error:
            print(f"{image_path}: ERROR: {error}")
            continue

        if not results:
            print(f"{image_path}: NO BARCODE DECODED")
            continue

        for index, result in enumerate(results, start=1):
            total += 1
            print(
                f"{image_path}: barcode {index}: {result['value']} "
                f"(format: {result['format']})"
            )

    print(f"\nDecoded {total} barcode(s) from {len(images)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
