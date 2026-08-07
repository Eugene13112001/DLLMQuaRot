"""Writing results out as CSV.

JSON keeps the nested structure, CSV is what actually gets opened in a
spreadsheet or read by pandas on a cluster.  Both are written, from the same
data, so they cannot drift apart.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Iterable, List, Optional, Sequence


def write_csv(
    path: str | pathlib.Path,
    rows: Sequence[dict],
    columns: Optional[Sequence[str]] = None,
) -> pathlib.Path:
    """Write ``rows`` as CSV; missing keys become empty cells.

    Column order is taken from ``columns`` when given, otherwise from the order
    keys first appear across the rows -- stable, so diffing two runs works.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen

    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return p


def sibling_csv(json_path: str | pathlib.Path) -> pathlib.Path:
    """out/compare.json -> out/compare.csv"""
    p = pathlib.Path(json_path)
    return p.with_suffix(".csv")


def round_floats(row: dict, digits: int = 6) -> dict:
    """Keep CSV cells readable without losing meaningful precision."""
    out = {}
    for k, v in row.items():
        out[k] = round(v, digits) if isinstance(v, float) else v
    return out


__all__ = ["write_csv", "sibling_csv", "round_floats"]
