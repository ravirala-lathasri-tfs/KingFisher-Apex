"""Quality-analysis adapter used during incremental migration."""

from typing import Any

import numpy as np

from barcode_preprocessor import analyze_image_quality as _analyze


def analyze_image_quality(frame: np.ndarray | None) -> dict[str, Any]:
    return _analyze(frame)
