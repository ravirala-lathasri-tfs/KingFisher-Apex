"""Format-agnostic pyzbar decoder adapter."""

from typing import Any

import numpy as np

from barcode_decoder import decode_barcodes


def decode(image: np.ndarray | None) -> list[dict[str, Any]]:
    return decode_barcodes(image)
