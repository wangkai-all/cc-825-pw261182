"""Probability-weighted tail risk summaries."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def weighted_quantile_value(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    pairs = sorted((float(v), max(0.0, float(w))) for v, w in zip(values, weights))
    total = math.fsum(w for _, w in pairs)
    if total <= 0:
        return pairs[-1][0] if pairs else 0.0
    target = min(1.0, max(0.0, alpha)) * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target - 1e-15:
            return v
    return pairs[-1][0]


def weighted_cvar(values: Sequence[float], weights: Sequence[float], alpha: float = 0.95) -> float:
    """Upper-tail conditional expectation using fractional mass at the boundary."""
    pairs = sorted(((float(v), max(0.0, float(w))) for v, w in zip(values, weights)), reverse=True)
    total = math.fsum(w for _, w in pairs)
    if total <= 0:
        return max(values) if values else 0.0
    tail_mass = min(total, max(0.0, 1.0 - alpha) * total)
    if tail_mass <= 1e-15:
        return pairs[0][0]

    remaining = tail_mass
    mass = 0.0
    acc = 0.0
    for v, w in pairs:
        take = min(w, remaining)
        acc += take * v
        mass += take
        remaining -= take
        if remaining <= 1e-15:
            break
    return acc / mass if mass > 0 else pairs[0][0]


def risk_summary(
    losses: Sequence[float],
    weights: Sequence[float],
    alpha: float = 0.95,
    *,
    ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    total_w = math.fsum(max(0.0, float(w)) for w in weights)
    if total_w <= 0:
        return {"expected": 0.0, "var": 0.0, "cvar": 0.0, "worst": 0.0, "tailMass": 0.0, "alpha": alpha}

    expected = math.fsum(float(v) * max(0.0, float(w)) for v, w in zip(losses, weights)) / total_w
    var = weighted_quantile_value(losses, weights, alpha)
    cvar = weighted_cvar(losses, weights, alpha)
    worst_idx = max(range(len(losses)), key=lambda i: losses[i])
    tail_mass = math.fsum(
        max(0.0, float(w)) for v, w in zip(losses, weights) if float(v) >= var - 1e-12
    ) / total_w
    return {
        "alpha": alpha,
        "expected": expected,
        "var": var,
        "cvar": cvar,
        "worst": float(losses[worst_idx]),
        "worstId": ids[worst_idx] if ids is not None else str(worst_idx),
        "tailMassAtThreshold": tail_mass,
        "totalProbabilityMass": total_w,
    }


def loss_scale(losses: Sequence[float], weights: Sequence[float]) -> float:
    """Robust scale used to put loss mismatch on a comparable footing to distance."""
    if not losses:
        return 1.0
    pairs = sorted((float(v), max(0.0, float(w))) for v, w in zip(losses, weights))
    vals = [v for v, _ in pairs]
    q10 = weighted_quantile_value(losses, weights, 0.10)
    q90 = weighted_quantile_value(losses, weights, 0.90)
    scale = q90 - q10
    if scale <= 1e-9:
        mean = sum(vals) / len(vals)
        scale = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
    if scale <= 1e-9:
        scale = max(1.0, max(vals) - min(vals))
    return max(1.0, scale)
