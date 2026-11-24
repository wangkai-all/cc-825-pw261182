"""Deterministic fictional datasets, including rare costly tail branches."""
from __future__ import annotations

import random
from typing import Any, Dict, List


DIMENSIONS = [
    {"key": "demand", "label": "区域需求", "unit": "千件", "scale": 10.0, "weight": 1.0},
    {"key": "freight", "label": "应急运价", "unit": "千元/千件", "scale": 3.0, "weight": 0.8},
    {"key": "disruption", "label": "中断强度", "unit": "级", "scale": 1.0, "weight": 1.2},
]

PERIODS = [f"第{t}周" for t in range(1, 7)]

PLANNING = {
    "orderCost": 10.5,
    "holdingCost": 1.2,
    "salvageValue": 3.2,
    "emergencyBaseCost": 13.5,
    "freightCostWeight": 1.15,
    "disruptionPenalty": 7.0,
    "qStep": 5.0,
    "tailAlpha": 0.95,
}


def _allocate_counts(n: int, masses: Dict[str, float]) -> Dict[str, int]:
    raw = {g: n * m for g, m in masses.items()}
    counts = {g: max(1 if n >= len(masses) else 0, int(x)) for g, x in raw.items()}
    # Ensure rare severe tail exists even in small demos.
    if n >= 8:
        counts["severe"] = max(2, counts["severe"])
    while sum(counts.values()) < n:
        g = max(raw, key=lambda x: raw[x] - counts[x])
        counts[g] += 1
    while sum(counts.values()) > n:
        g = max((g for g, c in counts.items() if c > 1), key=lambda x: counts[x])
        counts[g] -= 1
    return counts


def generate_fictional_dataset(n: int = 360, seed: int = 2611825, include_duplicates: bool = True) -> Dict[str, Any]:
    """Generate weighted multi-period paths from a hidden branching regime tree."""
    n = max(16, int(n))
    rng = random.Random(seed)
    group_mass = {"normal": 0.70, "high": 0.21, "storm": 0.075, "severe": 0.015}
    counts = _allocate_counts(n, group_mass)

    demand_base = [82.0, 86.0, 91.0, 96.0, 92.0, 88.0]
    freight_base = [6.0, 6.4, 6.9, 7.3, 7.0, 6.6]
    demand_add = {
        "normal": [0.0, 4.0, 7.0, 9.0, 6.0, 3.0],
        "high": [0.0, 18.0, 27.0, 34.0, 28.0, 20.0],
        "storm": [0.0, 32.0, 48.0, 58.0, 50.0, 38.0],
        "severe": [0.0, 32.0, 64.0, 80.0, 72.0, 56.0],
    }
    freight_add = {
        "normal": [0.0, 0.2, 0.4, 0.5, 0.3, 0.2],
        "high": [0.0, 1.8, 2.6, 3.2, 2.7, 1.8],
        "storm": [0.0, 3.5, 5.0, 6.2, 5.4, 4.0],
        "severe": [0.0, 3.5, 7.0, 9.4, 8.2, 6.1],
    }
    disruption = {
        "normal": [0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
        "high": [0.0, 0.4, 0.7, 0.6, 0.3, 0.1],
        "storm": [0.0, 1.6, 2.4, 2.0, 1.2, 0.5],
        "severe": [0.0, 1.6, 3.9, 4.7, 3.5, 1.9],
    }

    paths: List[Dict[str, Any]] = []
    serial = 1
    for group, count in counts.items():
        raw_weights = [rng.uniform(0.55, 1.55) for _ in range(count)]
        wsum = sum(raw_weights)
        for j in range(count):
            persistent_demand = rng.gauss(0.0, 5.0)
            values = []
            for t in range(6):
                if t <= 2:
                    d_noise = 0.0
                    f_noise = 0.0
                    x_noise = 0.0
                else:
                    d_noise = persistent_demand * 0.65 + rng.gauss(0.0, 3.8)
                    f_noise = rng.gauss(0.0, 0.35)
                    x_noise = max(-disruption[group][t] * 0.25, rng.gauss(0.0, 0.35))
                demand = max(1.0, demand_base[t] + demand_add[group][t] + d_noise)
                freight = max(0.5, freight_base[t] + freight_add[group][t] + f_noise)
                x = max(0.0, disruption[group][t] + x_noise)
                values.append([round(demand, 2), round(freight, 3), round(x, 3)])
            paths.append({
                "id": f"S{serial:05d}",
                "probability": group_mass[group] * raw_weights[j] / wsum,
                "values": values,
                "tags": [group, "fictional"],
            })
            serial += 1

    if include_duplicates and len(paths) >= 12:
        # Split existing probability mass into exact duplicate IDs.  Duplicate
        # merging must restore the original row/mass without changing total.
        for dup_i, source_index in enumerate([1, max(2, n // 3), max(3, 2 * n // 3)]):
            source = paths[source_index]
            half = source["probability"] / 2.0
            source["probability"] = half
            paths.append({
                "id": f"DUP{dup_i + 1}-{source['id']}",
                "probability": half,
                "values": source["values"],
                "tags": list(source.get("tags", [])) + ["imported-duplicate"],
            })

    return {
        "name": f"虚构多期应急需求情景（{len(paths)} 条导入行）",
        "description": "由隐藏的正常/高需求/风暴/严重中断分支生成；严重尾部概率低但损失大。",
        "periods": PERIODS,
        "dimensions": DIMENSIONS,
        "planning": PLANNING,
        "paths": paths,
        "metadata": {"generator": "scenario_tool.sample", "seed": seed, "basePathCount": n},
    }


def toy_dataset() -> Dict[str, Any]:
    """Small prefix-rich instance used by exhaustive-search tests and the UI."""
    dims = [
        {"key": "demand", "label": "需求", "unit": "千件", "scale": 10.0, "weight": 1.0},
        {"key": "freight", "label": "运价", "unit": "千元/千件", "scale": 3.0, "weight": 0.8},
        {"key": "disruption", "label": "中断", "unit": "级", "scale": 1.0, "weight": 1.2},
    ]
    periods = ["T1", "T2", "T3"]

    def row(d: float, f: float, x: float) -> List[float]:
        return [d, f, x]

    paths = [
        {"id": "A1", "probability": 0.22, "tags": ["normal"], "values": [row(10, 6, 0), row(20, 6.5, 0), row(28, 7.0, 0.1)]},
        {"id": "A2", "probability": 0.18, "tags": ["normal"], "values": [row(10, 6, 0), row(20, 6.5, 0), row(34, 7.2, 0.2)]},
        {"id": "B1", "probability": 0.16, "tags": ["high"], "values": [row(10, 6, 0), row(42, 8.0, 0.8), row(48, 8.5, 1.0)]},
        {"id": "B2", "probability": 0.12, "tags": ["high"], "values": [row(10, 6, 0), row(42, 8.0, 0.8), row(55, 9.1, 1.3)]},
        {"id": "C1", "probability": 0.07, "tags": ["storm"], "values": [row(10, 6, 0), row(75, 10.5, 2.0), row(82, 11.5, 2.8)]},
        {"id": "C2", "probability": 0.03, "tags": ["severe", "tail"], "values": [row(10, 6, 0), row(75, 10.5, 2.0), row(98, 13.0, 5.0)]},
        {"id": "D1", "probability": 0.13, "tags": ["normal"], "values": [row(24, 6.2, 0), row(30, 6.8, 0), row(33, 7.1, 0)]},
        {"id": "D2", "probability": 0.09, "tags": ["high"], "values": [row(24, 6.2, 0), row(30, 6.8, 0), row(44, 8.0, 0.6)]},
    ]
    return {
        "name": "小型前缀对照实例",
        "description": "8 paths with explicit common prefixes and one severe tail path.",
        "periods": periods,
        "dimensions": dims,
        "planning": {"qStep": 2.0, "orderCost": 10.5, "holdingCost": 1.2, "salvageValue": 3.2,
                     "emergencyBaseCost": 13.5, "freightCostWeight": 1.15, "disruptionPenalty": 7.0},
        "paths": paths,
    }
