from pathlib import Path

from decode_code128 import decode_code128


IMAGE = Path(__file__).parent / "Dataset" / "barcode (7).png"


def test_barcode_image_exists():
    assert IMAGE.exists(), f"Test image not found: {IMAGE}"


def test_decode_code128_barcode():
    results = decode_code128(str(IMAGE))
    assert results, "No Code 128 barcode was decoded"


if __name__ == "__main__":
    test_barcode_image_exists()
    test_decode_code128_barcode()
    print("PASS: barcode (7).png decoded successfully")
