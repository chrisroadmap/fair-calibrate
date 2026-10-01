"""Repository root, so scripts run from any working directory."""

from pathlib import Path

# src/fair_calibrate/paths.py -> repository root
ROOT = Path(__file__).resolve().parents[2]
