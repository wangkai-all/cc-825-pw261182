"""Business-scaled, chronologically ordered path distances."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * min(1.0, max(0.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo))


def derive_scales(ds: Dict[str, Any], mode: str = "metadata") -> List[float]:
    d_count = len(ds["dimensions"])
    t_count = len(ds["periods"])
    scales: List[float] = []
    for d, dim in enumerate(ds["dimensions"]):
        provided = dim.get("scale")
        if mode == "metadata" and _positive(provided):
            scales.append(float(provided))
            continue
        vals = [p["values"][t][d] for p in ds["paths"] for t in range(t_count)]
        vals.sort()
        scale = 0.0
        if mode in ("iqr", "metadata"):
            q1 = quantile(vals, 0.25)
            q3 = quantile(vals, 0.75)
            scale = q3 - q1
        if scale <= 1e-12:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
            scale = math.sqrt(var)
        if scale <= 1e-12:
            scale = max(vals) - min(vals)
        if scale <= 1e-12:
            scale = 1.0
        scales.append(scale)
    return scales


def _positive(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 1e-12
    except (TypeError, ValueError):
        return False


def common_prefix_length(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], decimals: int = 8) -> int:
    n = min(len(a), len(b))
    for t in range(n):
        if tuple(round(float(x), decimals) if decimals is not None else float(x) for x in a[t]) != tuple(
            round(float(x), decimals) if decimals is not None else float(x) for x in b[t]
        ):
            return t
    return n


class DistanceModel:
    """Weighted stage/prefix Euclidean distance using business scales."""

    def __init__(self, ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        options = options or {}
        self.ds = ds
        self.n = len(ds["paths"])
        self.t = len(ds["periods"])
        self.d = len(ds["dimensions"])
        self.scales = derive_scales(ds, str(options.get("scaleMode", "metadata")))

        custom_w = options.get("dimensionWeights") or {}
        weights = []
        for idx, dim in enumerate(ds["dimensions"]):
            if dim["key"] in custom_w:
                w = float(custom_w[dim["key"]])
            else:
                w = float(dim.get("weight", 1.0))
            weights.append(max(0.0, w))
        wsum = sum(weights) or 1.0
        self.dim_weights = [w / wsum for w in weights]

        curve = str(options.get("timeCurve", "early"))
        discount = float(options.get("temporalDiscount", 0.88))
        importance = []
        for t in range(self.t):
            if curve == "uniform":
                importance.append(1.0)
            elif curve == "late":
                importance.append(discount ** (self.t - 1 - t))
            else:  # early decisions/prefixes are more important
                importance.append(discount ** t)

        metric = str(options.get("distanceMetric", options.get("metric", "prefix")))
        if metric == "chronological":
            raw_w = importance
        else:
            # A difference at stage s belongs to every prefix ending at u >= s.
            # Therefore early divergence is charged repeatedly, which is the
            # metric-level reason common prefixes must not be split early.
            raw_w = [importance[s] * sum(importance[s:]) for s in range(self.t)]
        total_w = sum(raw_w) or 1.0
        self.stage_weights = [w / total_w for w in raw_w]
        self.metric = metric

        # Flatten scaled values for tight loops.
        self.flat: List[List[float]] = []
        for p in ds["paths"]:
            row: List[float] = []
            for tt in range(self.t):
                for dd in range(self.d):
                    row.append(float(p["values"][tt][dd]) / self.scales[dd])
            self.flat.append(row)

    def distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        a = self.flat[i]
        b = self.flat[j]
        total = 0.0
        stride = self.d
        for tt in range(self.t):
            stage = 0.0
            base = tt * stride
            for dd in range(self.d):
                delta = a[base + dd] - b[base + dd]
                stage += self.dim_weights[dd] * delta * delta
            total += self.stage_weights[tt] * stage
        return math.sqrt(max(0.0, total))

    def pairwise_to(self, indices: Sequence[int], reps: Sequence[int]) -> List[List[float]]:
        return [[self.distance(i, r) for r in reps] for i in indices]

    def full_matrix(self) -> List[List[float]]:
        out = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dij = self.distance(i, j)
                out[i][j] = dij
                out[j][i] = dij
        return out
