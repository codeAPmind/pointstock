"""Pre-market stock analysis system (milestone M1: data + GARCH range report)."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.json"
    with open(path) as f:
        cfg = json.load(f)
    # Resolve relative paths against the project root.
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg
