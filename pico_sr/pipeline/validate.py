from __future__ import annotations

import math
from typing import Any


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def validate_extraction_payload(payload: dict[str, Any]) -> list[str]:
    """Programmatic consistency checks; return human-readable flag strings."""
    flags: list[str] = []
    es = _num(payload.get("effect_size"))
    lo = _num(payload.get("ci_lower"))
    hi = _num(payload.get("ci_upper"))
    if es is not None and lo is not None and hi is not None:
        if lo > hi:
            flags.append("ci_lower_gt_ci_upper")
        elif not (lo <= es <= hi):
            flags.append("effect_size_outside_ci")
    pv = _num(payload.get("p_value"))
    if pv is not None:
        if pv < 0 or pv > 1:
            flags.append("p_value_out_of_0_1_range")
    n = _num(payload.get("population_n"))
    if n is not None and n <= 0:
        flags.append("population_n_non_positive")
    return flags
