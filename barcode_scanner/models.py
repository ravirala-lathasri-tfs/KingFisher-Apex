"""Typed data models shared by scanner subsystems."""

from dataclasses import dataclass, field
from typing import Any


Point = tuple[float, float]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageQuality:
    status: str
    problems: tuple[str, ...] = ()
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    dynamic_range: float = 0.0
    noise: float = 0.0
    edge_strength: float = 0.0


@dataclass
class BarcodeROI:
    polygon: list[Point]
    bbox: BBox
    barcode_family: str = "UNKNOWN"
    confidence: float = 0.0
    source: str = "unknown"
    roi_id: str | None = None


@dataclass(frozen=True)
class PreprocessingCandidate:
    name: str
    image: Any


@dataclass(frozen=True)
class PreprocessingDecision:
    problems: tuple[str, ...] = ()
    selected_operations: tuple[str, ...] = ()


@dataclass
class DecodeResult:
    success: bool
    data: str | None = None
    barcode_type: str | None = None
    roi_id: str | None = None
    source: str | None = None
    bbox: BBox | None = None


@dataclass
class BarcodeDetection:
    data: str
    barcode_type: str
    bbox: BBox | None = None
    polygon: list[Point] = field(default_factory=list)
    source: str = "unknown"


@dataclass
class ScanResult:
    detections: list[BarcodeDetection] = field(default_factory=list)
    failed_regions: list[BarcodeROI] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
