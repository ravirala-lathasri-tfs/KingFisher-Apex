"""Each transform starts from the unchanged source image."""
from collections.abc import Iterator
import cv2
import numpy as np
from .config import PipelineConfig
from .image_ops import adaptive_threshold, clahe_otsu, crop_1d_candidates, curved_1d_candidates, denoise_image, enhance_contrast, gamma_correct, otsu_threshold, perspective_candidates, resize_image, rotate_image, sharpen_for_barcode

def generate_variants(frame: np.ndarray, config: PipelineConfig) -> Iterator[tuple[str, np.ndarray]]:
    def experimental():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        background = cv2.GaussianBlur(gray, (0, 0), 25)
        normalized = cv2.divide(gray, background, scale=255)
        yield "experimental_illumination", normalized
        yield "experimental_illumination_otsu", cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        for sigma in (1.0, 2.0):
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            yield f"experimental_unsharp_{sigma:g}", cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
        # Wiener filtering with modest regularization limits noise amplification.
        def wiener(kernel):
            psf = np.zeros(gray.shape, np.float32)
            kh, kw = kernel.shape
            if kh > psf.shape[0] or kw > psf.shape[1]:
                return None
            psf[:kh, :kw] = kernel
            psf = np.roll(psf, (-(kh // 2), -(kw // 2)), axis=(0, 1))
            transfer = np.fft.fft2(psf)
            restored = np.fft.ifft2(np.fft.fft2(gray.astype(np.float32)) * np.conj(transfer) / (np.abs(transfer) ** 2 + 0.03)).real
            return np.clip(restored, 0, 255).astype(np.uint8)
        for length in (5, 9):
            for angle in (0, 45, 90, 135):
                kernel = np.zeros((length, length), np.uint8)
                center = (length - 1) / 2
                dx, dy = np.cos(np.deg2rad(angle)) * center, np.sin(np.deg2rad(angle)) * center
                cv2.line(kernel, (round(center-dx), round(center-dy)), (round(center+dx), round(center+dy)), 1, 1)
                restored = wiener(kernel.astype(np.float32) / kernel.sum())
                if restored is not None:
                    yield f"experimental_motion_{length}_{angle}", restored
        # QR detector points provide an independent rectification route.
        detector = cv2.QRCodeDetector()
        try:
            found, points = detector.detect(gray)
        except cv2.error:
            found, points = False, None
        if found and points is not None:
            points = np.asarray(points, dtype=np.float32).reshape(4, 2)
            side = max(64, min(1200, int(max(np.linalg.norm(points[(i+1)%4]-points[i]) for i in range(4)))))
            dest = np.float32([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]])
            rectified = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(points, dest), (side, side), borderValue=255)
            for scale in (1, 2):
                candidate = cv2.resize(rectified, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else rectified
                pad = max(8, candidate.shape[0] // 10)
                yield f"experimental_qr_rectified_pad_{scale}x", cv2.copyMakeBorder(candidate, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    def colleague_variants():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.fastNlMeansDenoising(clahe, None, 8, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        yield "clahe_denoise_sharpen", sharpened
        size = min(gray.shape[:2])
        block = min(31, size if size % 2 else size - 1)
        if block >= 3:
            yield "clahe_adaptive_gaussian", cv2.adaptiveThreshold(
                clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5
            )

    def rotate_bound(angle):
        h, w = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w, out_h = int(np.ceil(h * sine + w * cosine)), int(np.ceil(h * cosine + w * sine))
        matrix[0, 2] += (out_w - w) / 2
        matrix[1, 2] += (out_h - h) / 2
        return cv2.warpAffine(frame, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    operations = {
        "experimental": experimental,
        "equalize_hist": lambda: [("equalize_hist", cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame))],
        "combined_enhancement": colleague_variants,
        "wide_rotate": lambda: [(f"wide_rotate_{angle:+d}", rotate_bound(angle)) for step in range(5, 46, 5) for angle in (step, -step)],
        "resize_2x": lambda: [("resize_2x", resize_image(frame, 2.0))],
        "clahe": lambda: [("clahe", enhance_contrast(frame))],
        "otsu": lambda: [("otsu", otsu_threshold(frame))],
        "adaptive": lambda: [("adaptive", adaptive_threshold(frame))],
        "denoise": lambda: [("denoise", denoise_image(frame))],
        "sharpen": lambda: [("sharpen", sharpen_for_barcode(frame))],
        "crop_1d": lambda: crop_1d_candidates(frame),
        "invert": lambda: [("invert", cv2.bitwise_not(frame))],
        "clahe_otsu": lambda: [("clahe_otsu", clahe_otsu(frame))],
        "gamma": lambda: [(f"gamma_{gamma:g}", gamma_correct(frame, gamma)) for gamma in (0.7, 1.4)],
        "rotate": lambda: [(f"rotate_{angle:+d}", rotate_image(frame, angle)) for angle in (-5, 5, -10, 10)],
        "perspective": lambda: perspective_candidates(frame),
        "curved_1d": lambda: curved_1d_candidates(frame),
    }
    for name in config.variants:
        if name not in operations:
            raise ValueError(f"Unknown preprocessing variant: {name}")
        yield from operations[name]()
