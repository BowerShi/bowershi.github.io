"""IO utilities for storing FEniCS fields and logs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from dolfin import Function, XDMFFile


def save_function_xdmf(path: Path, function: Function, name: str = "field") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(str(path)) as xdmf:
        xdmf.write(function, name)


def save_history_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow([f"{value:.16e}" for value in row])
