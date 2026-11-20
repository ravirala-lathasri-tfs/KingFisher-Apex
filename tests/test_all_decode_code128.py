import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

from decode_code128 import decode_code128


ROOT = Path(__file__).parent / "Dataset"
OUTPUT = Path(__file__).parent / "code128_batch_results.csv"


def main():
    rows = []
    for number in range(1, 46):
        path = ROOT / f"barcode ({number}).png"
        if not path.exists():
            rows.append([path.name, "MISSING", "", ""])
            continue
        # Suppress the single-image function's verbose output during batch testing.
        with redirect_stdout(io.StringIO()):
            results = decode_code128(str(path))
        if results:
            rows.append([path.name, "DECODED", "CODE128", results])
        else:
            rows.append([path.name, "NOT_DECODED", "", ""])

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["filename", "status", "barcode_type", "decoded_value"], *rows])

    decoded = sum(row[1] == "DECODED" for row in rows)
    missing = sum(row[1] == "MISSING" for row in rows)
    failed = len(rows) - decoded - missing
    print(f"Total requested: {len(rows)}")
    print(f"Decoded: {decoded}")
    print(f"Not decoded: {failed}")
    print(f"Missing: {missing}")
    print(f"Results saved to: {OUTPUT}")
    for filename, status, barcode_type, value in rows:
        print(f"{filename}: {status}" + (f" | {barcode_type} | {value}" if status == "DECODED" else ""))


if __name__ == "__main__":
    main()
