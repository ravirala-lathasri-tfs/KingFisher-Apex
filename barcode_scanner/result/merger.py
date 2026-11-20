"""Result-merging adapter."""

from typing import Any

from barcode_decoder import merge_unique_physical_detections


def merge(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_unique_physical_detections(*groups)
