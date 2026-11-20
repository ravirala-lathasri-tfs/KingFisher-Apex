"""Preprocess one image and try each variant with Pyzbar/ZBar.

Run from the project root:
    .venv311\\Scripts\\python.exe .\\preprocess_then_pyzbar.py "C:\\path\\to\\average_image.jpg"
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import cv2
from pyzbar.pyzbar import decode

from preprocessing.config import PipelineConfig
from preprocessing.variants import generate_variants


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess one image and decode each variant with Pyzbar."
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
            results = decode(candidate)
        except Exception as error:
            print(f"Pyzbar error on {method}: {type(error).__name__}: {error}")
            continue

        if results:
            print(f"Decoded after preprocessing: {method}")
            for index, result in enumerate(results, start=1):
                value = result.data.decode("utf-8", errors="replace")
                print(f"Barcode {index}: {value} (type: {result.type})")
            return 0

    print(f"No barcode decoded after {attempts} preprocessing attempt(s).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
