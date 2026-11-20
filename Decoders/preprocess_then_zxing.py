"""Preprocess one image and try each variant with the ZXing-C++ decoder.

Run from the project root:
    .venv311\\Scripts\\python.exe .\\preprocess_then_zxing.py "C:\\path\\to\\average_image.jpg"
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import zxingcpp
from PIL import Image

from preprocessing.config import PipelineConfig
from preprocessing.variants import generate_variants


def decode_with_zxing(frame: np.ndarray) -> list:
    """Decode an OpenCV BGR or grayscale image with ZXing-C++."""
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return zxingcpp.read_barcodes(Image.fromarray(rgb))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess one image and decode each variant with ZXing-C++."
    )
    parser.add_argument("image", type=Path, help="Path to one image to decode")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=13,
        help="Maximum preprocessing variants to try (default: 13)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        print(f"Could not read image: {args.image}")
        return 1

    config = PipelineConfig(max_attempts=max(1, args.max_attempts))
    attempts = 0
    for attempts, (method, candidate) in enumerate(
        islice(generate_variants(frame, config), config.max_attempts), start=1
    ):
        try:
            results = decode_with_zxing(candidate)
        except Exception as error:
            print(f"ZXing error on {method}: {type(error).__name__}: {error}")
            continue

        if results:
            print(f"Decoded after preprocessing: {method}")
            for index, result in enumerate(results, start=1):
                print(f"Barcode {index}: {result.text} (format: {result.format})")
            return 0

    print(f"No barcode decoded after {attempts} preprocessing attempt(s).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
