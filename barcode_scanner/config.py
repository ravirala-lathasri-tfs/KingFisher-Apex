"""Configuration for the modular scanner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    output_dir: str = "test_output"


@dataclass(frozen=True)
class ScannerConfig:
    enable_qr: bool = True
    enable_1d: bool = True
    max_recovery_attempts: int = 8
    debug: DebugConfig = DebugConfig()
