#!/usr/bin/env python
"""Read-only helper: report candidate entity/time columns for a dataset.

Loads a .csv/.parquet, infers a data dictionary with ``expdpy.build_data_def``, and prints
the columns expdpy would treat as the panel **entity** (unit) and **time** dimensions, plus
each column's dtype and cardinality. It writes nothing - it is a planning aid only.

Usage::

    python .claude/skills/use-expdpy/scripts/check_panel.py path/to/data.csv
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Print candidate entity/time columns and per-column dtypes for a data file."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2

    path = Path(args[0]).expanduser()
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    import pandas as pd

    import expdpy as xp

    df = (
        pd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )
    data_def = xp.build_data_def(df)

    by_type: dict[str, list[str]] = {}
    for _, row in data_def.iterrows():
        by_type.setdefault(str(row["type"]), []).append(str(row["var_name"]))

    print(f"{path.name}: {len(df)} rows x {df.shape[1]} columns\n")
    print("Inferred panel dimensions:")
    print(f"  entity (unit): {by_type.get('entity', []) or '(none inferred)'}")
    print(f"  time         : {by_type.get('time', []) or '(none inferred)'}")
    print(
        "\nDeclare the panel with:\n"
        "  df = xp.set_panel(df, entity='<entity>', time='<time>')\n"
    )
    print("Columns (name | dtype | n_unique):")
    for col in df.columns:
        print(f"  {col} | {df[col].dtype} | {df[col].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
