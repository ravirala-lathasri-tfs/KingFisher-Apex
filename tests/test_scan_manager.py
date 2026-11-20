from processing.scan_manager import ScanManager

class Provider:
    def get_sample_information(self, code):
        if code == "UNKNOWN": return {"barcode": code, "sample_id": None, "sample_type": "UNKNOWN"}
        return {"barcode": code, "sample_id": code, "sample_type": "ALCOHOL" if code == "BAD" else "WATER"}

def test_processing_and_duplicates():
    m = ScanManager(Provider())
    assert m.process("A")["result"] == "PASS"
    assert m.process("A") is None
    assert m.processed_count == 1

def test_many_unknown_and_reset():
    m = ScanManager(Provider())
    for i in range(15): assert m.process(f"S{i}")
    assert m.processed_count == 15
    assert m.process("BAD")["result"] == "FAIL"
    assert m.process("UNKNOWN")["result"] == "UNRESOLVED"
    m.reset_session(); assert m.processed_count == 0 and not m.results
