#!/usr/bin/env python3
"""Raspberry Pi test: type PHASEON to capture an image and run another decoder.

This script owns camera capture and saves each frame to an image file. It then
imports decode_code128.py and calls decode_code128(image_path). The decoder's
stdout/stderr is shown in this terminal.
It does not open the Pi UART or send decoded data to KingFisher.

For PuTTY, SSH into the Pi, run this program, then type PHASEON and press Enter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


def capture_image(camera: cv2.VideoCapture, image_path: Path) -> bool:
    """Capture one frame and save it for the separate decoder program."""
    ok, frame = camera.read()
    if not ok:
        print("ERROR: Camera frame capture failed.", flush=True)
        return False
    if not cv2.imwrite(str(image_path), frame):
        print(f"ERROR: Could not save captured image to {image_path}", flush=True)
        return False
    print(f"Captured image: {image_path}", flush=True)
    return True


def run_decoder(decoder_path: Path, image_path: Path) -> int:
    """Import decode_code128.py and call decode_code128(captured_image_path)."""
    if not decoder_path.is_file():
        print(f"ERROR: Decoder script not found: {decoder_path}", flush=True)
        return 2

    # Importing the file avoids its standalone __main__ block, which currently
    # points at a hard-coded Windows image. Call its function with our Pi image.
    runner = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('barcode_decoder', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "result = module.decode_code128(sys.argv[2]); "
        "print('\\nFinal result:', result)"
    )
    print(f"Running decode_code128() from: {decoder_path}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-c", runner, str(decoder_path), str(image_path)],
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"Decoder exited with code {completed.returncode}.", flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--decoder",
        type=Path,
        default=None,
        help="Path to decode_code128.py; defaults to the file beside this wrapper",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().with_name("captured_barcode.jpg"),
        help="Image file written on each PHASEON (default: captured_barcode.jpg beside this script)",
    )
    parser.add_argument("--width", type=int, default=0, help="Optional camera frame width")
    parser.add_argument("--height", type=int, default=0, help="Optional camera frame height")
    args = parser.parse_args()

    if args.decoder is None:
        script_dir = Path(__file__).resolve().parent
        args.decoder = script_dir / "decode_code128.py"

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}", file=sys.stderr)
        return 2

    if args.width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Camera ready. Type PHASEON and press Enter to capture, then run the decoder.", flush=True)
    print("Type QUIT to exit.", flush=True)

    try:
        for line in sys.stdin:
            command = line.strip().upper()
            if command == "QUIT":
                break
            if command != "PHASEON":
                print("Ignored. Enter PHASEON to trigger a scan.", flush=True)
                continue

            print("PHASEON received; capturing and decoding...", flush=True)
            if capture_image(camera, args.image):
                run_decoder(args.decoder.resolve(), args.image.resolve())
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()

    print("Camera test stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
