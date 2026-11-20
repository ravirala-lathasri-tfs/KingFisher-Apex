"""High-level scanner facade; algorithm migration is incremental."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import ScannerConfig
from .models import BarcodeDetection, ScanResult

logger = logging.getLogger(__name__)


class BarcodeScanner:
    """Public scanner facade preserving the existing decoder behavior."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()

    def scan(self, frame: np.ndarray | None) -> ScanResult:
        """Scan an image through the existing decoder during migration."""
        from barcode_decoder import decode_barcodes_with_recovery

        report: dict[str, Any] = decode_barcodes_with_recovery(frame)
        detections = [
            BarcodeDetection(
                data=item.get("decoded_data", ""),
                barcode_type=item.get("barcode_type", "UNKNOWN"),
                bbox=item.get("bbox"),
                polygon=item.get("polygon", []),
                source="original" if not item.get("recovered") else "recovered",
            )
            for item in report.get("results", [])
        ]
        logger.info("Scan complete: %d detections", len(detections))
        return ScanResult(detections=detections, diagnostics=report)
