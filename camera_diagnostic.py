"""Temporary raw-camera diagnostic; independent of the barcode scanner."""
import cv2

BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("CAP_ANY", cv2.CAP_ANY)]
INDICES = range(4)

def inspect_camera(index, backend_name, backend):
    cap = cv2.VideoCapture(index, backend)
    report = {"index": index, "backend": backend_name, "opened": bool(cap.isOpened()), "frame": None}
    if report["opened"]:
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size:
                report["frame"] = frame  # raw frame, untouched
    if report["frame"] is not None:
        frame = report["frame"]
        report.update(shape=frame.shape, dtype=str(frame.dtype), minimum=int(frame.min()),
                      maximum=int(frame.max()), mean=float(frame.mean()))
        report["valid"] = (report["maximum"] - report["minimum"] > 5 and report["mean"] > 1)
    else:
        report.update(shape=None, dtype=None, minimum=None, maximum=None, mean=None, valid=False)
    cap.release()
    return report

def print_report(report):
    print(f"\nCamera index: {report['index']}\nBackend: {report['backend']}\n"
          f"Opened: {report['opened']}\nFrame received: {report['frame'] is not None}\n"
          f"Shape: {report['shape']}\nDtype: {report['dtype']}\nMin: {report['minimum']}\n"
          f"Max: {report['maximum']}\nMean: {report['mean']:.1f}" if report['mean'] is not None else
          f"\nCamera index: {report['index']}\nBackend: {report['backend']}\nOpened: {report['opened']}\n"
          f"Frame received: False\nShape: None\nDtype: None\nMin: None\nMax: None\nMean: None")
    if report["frame"] is not None and not report["valid"]:
        print('Status: BLANK - Camera opened but returned blank frame.')
    elif report["valid"]:
        print('Status: VALID - Valid image frames received.')
    else:
        print('Status: NO FRAME')

def run_detection():
    selected = None
    for index in INDICES:
        for name, backend in BACKENDS:
            report = inspect_camera(index, name, backend)
            print_report(report)
            if report["valid"] and selected is None:
                selected = (index, name, backend)
    return selected

def main():
    while True:
        selected = run_detection()
        if selected is None:
            print("\nNo camera/backend combination produced a valid non-black image.")
            print("Press R to retry or Q to quit.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"): break
            continue
        index, name, backend = selected
        print(f"\nSELECTED: camera index {index}, backend {name}")
        cap = cv2.VideoCapture(index, backend)
        while cap.isOpened():
            ret, frame = cap.read()  # display the raw frame exactly as returned
            if not ret or frame is None:
                print("Selected stream stopped returning frames.")
                break
            cv2.imshow("RAW CAMERA DIAGNOSTIC", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return
            if key == ord("r"):
                cap.release(); cv2.destroyWindow("RAW CAMERA DIAGNOSTIC"); break
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__": main()
