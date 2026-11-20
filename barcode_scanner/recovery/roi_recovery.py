"""ROI recovery adapter; full strategy migration remains incremental."""

from typing import Any

import numpy as np

from barcode_decoder import decode_barcodes_with_recovery


def recover(image: np.ndarray | None) -> dict[str, Any]:
    return decode_barcodes_with_recovery(image)
