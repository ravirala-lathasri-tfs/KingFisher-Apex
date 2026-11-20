#!/usr/bin/env python3
"""Camera barcode-reader replacement triggered over the Raspberry Pi UART.

UART protocol for PuTTY bring-up:
  Host -> Pi: raw ASCII PHASEON (no line ending required)
  Pi -> host: decoded barcode text followed by CRLF

The script captures one image and calls decode_code128(image_path) from the
neighboring decode_code128.py. It intentionally does not add P/T prefixes.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path

import serial


PHASE_ON = b"PHASEON"


def load_decoder(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Decoder script not found: {path}")
    spec = importlib.util.spec_from_file_location("kf_barcode_decoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import decoder script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "decode_code128", None)):
        raise AttributeError(f"{path} must define decode_code128(image_path)")
    return module.decode_code128


def capture_and_decode(image_path: Path, decode_image) -> str | None:
    """Capture a still with the Raspberry Pi camera stack, then decode it."""
    try:
        capture = subprocess.run(
            [
                "rpicam-still",
                "--nopreview",
                "--timeout",
                "1000",
                "--output",
                str(image_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.error("Could not run rpicam-still: %s", exc)
        return None
    if capture.returncode != 0:
        logging.error("rpicam-still failed: %s", capture.stderr.strip())
        return None

    logging.info("Captured %s; running decoder", image_path)
    result = decode_image(str(image_path))
    if result is None:
        return None
    value = str(result).strip()
    return value or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/serial0", help="Pi UART device (default: /dev/serial0)")
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path(__file__).resolve().with_name("decode_code128.py"),
        help="Path to the decoder script (default: decode_code128.py beside this file)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().with_name("captured_barcode.jpg"),
        help="Captured image path",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        decode_image = load_decoder(args.decoder.resolve())
    except (OSError, ImportError, AttributeError) as exc:
        logging.error("Decoder setup failed: %s", exc)
        return 2

    try:
        with serial.Serial(
            port=args.port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=2.0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as uart:
            logging.info("Listening on %s at 115200 8N1; waiting for PHASEON", args.port)
            recent = bytearray()
            while True:
                data = uart.read(64)
                if not data:
                    continue
                recent.extend(data)
                if len(recent) > len(PHASE_ON):
                    del recent[:-len(PHASE_ON)]
                if recent.endswith(PHASE_ON):
                    recent.clear()
                    logging.info("PHASEON received")
                    value = capture_and_decode(args.image, decode_image)

                    # Firmware can repeat PHASEON while waiting. Drop any
                    # repeated triggers accumulated during camera/decode work.
                    uart.reset_input_buffer()
                    if value is None:
                        uart.write(b"NO_BARCODE\r\n")
                        logging.info("No barcode decoded; sent NO_BARCODE")
                    else:
                        response = value.encode("utf-8", errors="replace") + b"\r\n"
                        uart.write(response)
                        uart.flush()
                        logging.info("Returned decoded text to host")
                    time.sleep(0.05)
    except serial.SerialException:
        logging.exception("Could not use UART %s", args.port)
        return 4
    except KeyboardInterrupt:
        logging.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
