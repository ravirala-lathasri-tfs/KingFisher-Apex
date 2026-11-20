from barcode.barcode_detector import classify_barcode_type


def test_barcode_categories():
    assert classify_barcode_type("EAN13") == "1D barcode"
    assert classify_barcode_type("CODE128") == "1D barcode"
    assert classify_barcode_type("QR_CODE") == "QR code"
    assert classify_barcode_type("DATA_MATRIX") == "2D barcode"
    assert classify_barcode_type("PDF417") == "2D barcode"
