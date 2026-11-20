from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass
class PipelineConfig:
    variants: tuple[str, ...] = ("resize_2x", "clahe", "otsu", "perspective", "curved_1d", "adaptive", "denoise", "sharpen", "crop_1d", "invert", "clahe_otsu", "gamma", "rotate", "equalize_hist", "combined_enhancement", "wide_rotate")
    max_attempts: int | None = None
    debug: bool = False
    output_dir: Path = Path("preprocessing_output")
    expected_csv: Path | None = None
