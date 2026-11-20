# Barcode preprocessing and classification

The package classifies images with Pyzbar using these rules:

- **Good:** the original image decodes.
- **Average:** the original fails, but a configured independent preprocessing variant decodes.
- **Bad:** all configured attempts fail. This means unrecovered by this pipeline.
- **UNREADABLE / ERROR:** image loading or processing failed; these are kept separate from Bad.

Decoded values are `UNVERIFIED` without expected values. Supply a CSV with `filename` (or `image_path`) and `expected_value` (or `expected`) columns to get `MATCH` or `MISMATCH` validation.

## Structure

```text
preprocessing/
  config.py       attempt order, attempt cap, paths, debug flag
  image_ops.py    adapters to existing transforms and region correction
  variants.py     independently generated preprocessing attempts
  decoder.py      Pyzbar adapter
  pipeline.py     per-image classification
  batch.py        CLI, copies, and CSV report
```

The existing `barcode_preprocessor.py` and `barcode_decoder.py` remain unchanged. `image_ops.py` reuses the existing CLAHE, Otsu/adaptive threshold, denoise, sharpen, resize, QR localization, and QR rectification functions. `decoder.py` imports the existing `decode_barcodes` Pyzbar implementation and supplements QR results by decoding detected QR regions independently. When multiple QR values are found, `decoded_value` in the report contains them separated by semicolons. The extra operations are independent rotation attempts and small adapter functions.

## Setup

From the project root, install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Pyzbar also needs a working ZBar runtime on the system; this repository configures it through `barcode_scanner.runtime`.

## Run only the existing Bad folder

```powershell
python -m preprocessing.batch --bad-dir "Dataset_Split\train\bad" --output "preprocessing_output\train_bad"
```

Output copies are written under `Average/`, `Bad/`, `UNREADABLE/`, and `ERROR/`. Source files are never moved or overwritten.

## Reclassify a full dataset

The dataset root can contain any number of nested directories. Every supported image under it is evaluated from its original, so prior folder names do not determine the new class:

```powershell
python -m preprocessing.batch --dataset "Dataset_Split\train" --output "preprocessing_output\train_all"
```

## Ground truth and options

```powershell
python -m preprocessing.batch --bad-dir "Dataset_Split\train\bad" --output "preprocessing_output\checked" --expected-csv "expected.csv" --max-attempts 8 --debug
```


The expected CSV needs a filename column and expected barcode value column; accepted aliases are documented above. Debug images are saved only for a successful processed candidate in `debug/`. The CSV report is `report.csv` and includes image path, class, decoded value, validation status, successful method, elapsed seconds, and errors.

The additional variants include detected-region perspective correction, cylindrical 1D barcode unwrapping with `remap`, 1D barcode region crops with quiet-zone padding, inversion, CLAHE followed by Otsu, and two gamma corrections. Curved 1D variants use OpenCV's `BarcodeDetector` to locate the barcode quadrilateral, rectify its long axis horizontally, then independently unwrap several curvature strengths. Pyzbar is tried first for each preprocessed candidate; OpenCV's barcode decoder is the fallback. This requires an OpenCV build that includes `cv2.barcode.BarcodeDetector`; other pipeline operations continue to work if that module is unavailable. Each candidate is generated from the original image. Increasing `--max-attempts` can reach later fallbacks, at the cost of longer processing.
Default variants run in order: resize, CLAHE, Otsu, adaptive threshold, denoise, sharpen, small-angle rotations, then detected QR perspective correction. Each starts from the original image and decoding stops at first success. Change `PipelineConfig.variants` in `config.py` to tune order; `max_attempts` bounds generated attempts. QR perspective correction is available only when the existing OpenCV/local contour localization finds a quadrilateral. Rotation angles are fixed in `variants.py`.
