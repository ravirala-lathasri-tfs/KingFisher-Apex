"""Modular barcode scanning public API."""

from .models import ScanResult
from .scanner import BarcodeScanner

__all__ = ["BarcodeScanner", "ScanResult"]
