# Camera Barcode Scanner

Install with `py -3.13 -m venv .venv`, activate the environment, then `pip install -r requirements.txt`.

Run with `python app.py`; press **Q** to quit or **R** to reset the session. Run tests with `pytest`.

For a file: `python test_barcode_images.py Dataset\\Google-image-6.png`. Add `--debug` to log every preprocessing attempt. Install on Windows with Python 3.12 or newer: `py -m venv .venv`, `.venv\\Scripts\\Activate.ps1`, then `python -m pip install -r requirements.txt`. Pyzbar is attempted first; if its native ZBar DLL cannot load, zxing-cpp is used automatically. Neither decoder uses OCR or filename values.

The camera adapter captures frames, `pyzbar` decodes barcodes, and `ScanManager` processes each unique value once. A decoded image is `GOOD`; an undecodable image is `BAD` and should be retaken. Decoder formats are normalized to `1D barcode`, `2D barcode`, or `QR code`. It looks up sample information in `data/sample_database.json`, compares its type with `config/config.json`, and appends results to `data/scan_results.csv`. No image classification is used. To replace JSON with a company API, implement `SampleProvider` and pass it to `ScanManager`; the camera and business logic remain unchanged.
