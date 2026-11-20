"""Preprocessing facade; legacy-compatible migration entry point."""

from barcode_preprocessor import (
    PreprocessConfig,
    adaptive_threshold,
    analyze_image_quality,
    apply_adaptive_preprocessing,
    describe_adaptive_candidates,
    denoise_image,
    enhance_contrast,
    generate_adaptive_candidates,
    generate_roi_candidates,
    order_quad_points,
    otsu_threshold,
    rectify_barcode_roi,
    rectify_qr,
    sharpen_for_barcode,
    sharpen_image,
    upscale_image,
)

__all__ = [
    "PreprocessConfig", "analyze_image_quality", "generate_adaptive_candidates",
    "describe_adaptive_candidates", "apply_adaptive_preprocessing",
    "enhance_contrast", "sharpen_image", "sharpen_for_barcode", "denoise_image",
    "upscale_image", "adaptive_threshold", "otsu_threshold", "order_quad_points",
    "rectify_qr", "rectify_barcode_roi", "generate_roi_candidates",
]
