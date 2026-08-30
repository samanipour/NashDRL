from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, **state: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)
