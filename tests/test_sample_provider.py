from sample_data.json_sample_provider import JsonSampleProvider

def test_json_lookup(tmp_path):
    db = tmp_path / "db.json"
    db.write_text('{"X": {"sample_id": "X", "sample_type": "WATER"}}', encoding="utf-8")
    p = JsonSampleProvider(db)
    assert p.get_sample_information("X")["sample_type"] == "WATER"
    assert p.get_sample_information("Y")["sample_type"] == "UNKNOWN"
