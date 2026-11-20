"""Platform-specific barcode runtime setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_barcode_runtime() -> None:
    """Configure native lookup only where the host platform requires it.

    Linux/Raspberry Pi uses the system ZBar installation and needs no DLL
    path handling here. Windows may need the directory containing pyzbar's
    native libraries added before importing pyzbar.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / ".venv311" / "Lib" / "site-packages" / "pyzbar",
        project_root / ".venv" / "Lib" / "site-packages" / "pyzbar",
    ]
    for directory in candidates:
        if directory.is_dir():
            os.add_dll_directory(str(directory))
            return
