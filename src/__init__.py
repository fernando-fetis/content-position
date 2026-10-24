from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN: dict = {}


@cache
def registry() -> dict[str, dict]:
    return yaml.safe_load((ROOT / 'src' / 'models.yaml').read_text())
