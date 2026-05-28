from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import pandas as pd
from scipy.io import arff


_COMPACT_ATTRIBUTE_RE = re.compile(
    r"^(\s*@attribute\s+)([^\s{]+)(\{.*\}\s*)$",
    flags=re.IGNORECASE,
)


def load_arff_dataframe(path: str | Path) -> pd.DataFrame:
    try:
        data, _ = arff.loadarff(str(path))
    except arff.ParseArffError:
        data, _ = _load_normalized_arff(path)

    df = pd.DataFrame(data)

    for col in df.columns:
        df[col] = df[col].map(_decode_arff_value)

    return df


def _load_normalized_arff(path: str | Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    fixed_lines = []
    changed = False

    for line in text.splitlines(keepends=True):
        fixed_line = _normalize_attribute_line(line)
        changed = changed or fixed_line != line
        fixed_lines.append(fixed_line)

    if not changed:
        return arff.loadarff(str(path))

    return arff.loadarff(StringIO("".join(fixed_lines)))


def _normalize_attribute_line(line: str) -> str:
    match = _COMPACT_ATTRIBUTE_RE.match(line)
    if match is None:
        return line

    prefix, name, values = match.groups()
    return f"{prefix}{name} {values}"


def _decode_arff_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
