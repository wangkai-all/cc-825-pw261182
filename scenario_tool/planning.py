"""A small, exact-enumerable multi-period inventory planning model.

Decision variables
------------------
Q : here-and-now initial stock/procurement quantity.

Scenario simulation
-------------------
Stock is consumed chronologically.  Demand that cannot be served from stock is
emergency-procured at a cost that rises with freight and disruption.  Leftover
stock incurs holding cost and has salvage value.  Thus temporal ordering and
tail disruptions affect the result.

The model is intentionally small enough to solve by exhaustive candidate-grid
or breakpoint enumeration with the standard library only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .risk import risk_summary

DEFAULT_PARAMS: Dict[str, Any] = {
    "orderCost": 10.5,
    "holdingCost": 1.2,
    "salvageValue": 3.2,
    "emergencyBaseCost": 13.5,
    "freightCostWeight": 1.15,
    "disruptionPenalty": 7.0,
    "demandDimension": "demand",
    "freightDimension": "freight",
    "disruptionDimension": "disruption",
    "qStep": 5.0,
    "qMin": 0.0,
    "qMax": None,
    "maxCandidateCount": 101,
    "refine": False,
}


def planning_parameters(ds: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update(ds.get("planning") or {})
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def dimension_index(ds: Dict[str, Any], key: str, fallback: int = 0) -> Optional[int]:
    keys = [d["key"] for d in ds["dimensions"]]
    if key in keys:
        return keys.index(key)
    # Tolerate common aliases.
    aliases = {
        "demand": ["demand", "volume", "load", "sales", "需求"],
        "freight": ["freight", "price", "cost", "fuel", "运价", "价格"],
        "disruption": ["disruption", "outage", "shock", "severity", "中断"],
    }.get(key, [])
    for alias in aliases:
        if alias in keys:
            return keys.index(alias)
    return fallback if 0 <= fallback < len(keys) else None


def simulate_path(
    values: Sequence[Sequence[float]],
    q: float,
    params: Dict[str, Any],
    dims: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, float]:
    if dims is None:
        dims = {"demand": 0, "freight": 1, "disruption": 2}
    i_demand = dims.get("demand")
    i_freight = dims.get("freight")
    i_disrupt = dims.get("disruption")

    stock = float(q)
    order_cost = float(params["orderCost"]) * q
    holding = 0.0
    emergency_cost = 0.0
    emergency_units = 0.0
    served = 0.0
    total_demand = 0.0

    for row in values:
        demand = float(row[i_demand]) if i_demand is not None and i_demand < len(row) else float(row[0])
        freight = float(row[i_freight]) if i_freight is not None and i_freight < len(row) else 0.0
        disruption = float(row[i_disrupt]) if i_disrupt is not None and i_disrupt < len(row) else 0.0
        total_demand += max(0.0, demand)
        used = min(max(stock, 0.0), max(0.0, demand))
        stock -= used
        served += used
        shortage = max(0.0, demand) - used
        emergency_units += shortage
        unit_emergency = (
            float(params["emergencyBaseCost"])
            + float(params["freightCostWeight"]) * freight
            + float(params["disruptionPenalty"]) * max(0.0, disruption)
        )
        emergency_cost += shortage * unit_emergency
        holding += float(params["holdingCost"]) * max(0.0, stock)

    salvage = float(params["salvageValue"]) * max(0.0, stock)
    total = order_cost + holding + emergency_cost - salvage
    return {
        "totalCost": total,
        "orderCost": order_cost,
        "holdingCost": holding,
        "emergencyCost": emergency_cost,
        "salvage": salvage,
        "emergencyUnits": emergency_units,
        "servedUnits": served,
        "totalDemand": total_demand,
        "leftover": max(0.0, stock),
        "serviceLevel": served / total_demand if total_demand > 1e-12 else 1.0,
    }


def _dim_map(ds: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Optional[int]]:
    return {
        "demand": dimension_index(ds, str(params["demandDimension"]), 0),
        "freight": dimension_index(ds, str(params["freightDimension"]), 1 if len(ds["dimensions"]) > 1 else 0),
        "disruption": dimension_index(ds, str(params["disruptionDimension"]), 2 if len(ds["dimensions"]) > 2 else 0),
    }


class FastPlanningCost:
    """Tight array-based cost evaluator used for exhaustive Q curves."""

    def __init__(self, ds: Dict[str, Any], params: Dict[str, Any]):
        dims = _dim_map(ds, params)
        i_demand = dims["demand"] or 0
        i_freight = dims["freight"]
        i_disrupt = dims["disruption"]
        self.order = float(params["orderCost"])
        self.holding = float(params["holdingCost"])
        self.salvage = float(params["salvageValue"])
        base_emerg = float(params["emergencyBaseCost"])
        fw = float(params["freightCostWeight"])
        dp = float(params["disruptionPenalty"])
        self.demand: List[List[float]] = []
        self.unit_emergency: List[List[float]] = []
        for p in ds["paths"]:
            d_row = []
            u_row = []
            for row in p["values"]:
                demand = max(0.0, float(row[i_demand])) if i_demand < len(row) else max(0.0, float(row[0]))
                freight = float(row[i_freight]) if i_freight is not None and i_freight < len(row) else 0.0
                disrupt = float(row[i_disrupt]) if i_disrupt is not None and i_disrupt < len(row) else 0.0
                d_row.append(demand)
                u_row.append(base_emerg + fw * freight + dp * max(0.0, disrupt))
            self.demand.append(d_row)
            self.unit_emergency.append(u_row)

    def path_cost(self, path_index: int, q: float) -> float:
        stock = float(q)
        cost = self.order * q
        demands = self.demand[path_index]
        units = self.unit_emergency[path_index]
        h = self.holding
        salvage = self.salvage
        for t in range(len(demands)):
            d = demands[t]
            if stock >= d:
                stock -= d
                shortage = 0.0
            else:
                shortage = d - stock
                stock = 0.0
            cost += shortage * units[t] + h * stock
        return cost - salvage * stock

    def expected_cost(self, q: float, weights: Sequence[float]) -> float:
        total_w = math.fsum(max(0.0, float(w)) for w in weights)
        if total_w <= 0:
            total_w = 1.0
        acc = 0.0
        for i, w0 in enumerate(weights):
            w = max(0.0, float(w0))
            if w:
                acc += w * self.path_cost(i, q)
        return acc / total_w


def path_costs(
    ds: Dict[str, Any],
    q: float,
    params: Optional[Dict[str, Any]] = None,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    params = planning_parameters(ds, params)
    dims = _dim_map(ds, params)
    return [simulate_path(p["values"], q, params, dims) for p in ds["paths"]]


def path_loss_values(ds: Dict[str, Any], q: float, params: Optional[Dict[str, Any]] = None) -> List[float]:
    params = planning_parameters(ds, params)
    fast = FastPlanningCost(ds, params)
    return [fast.path_cost(i, q) for i in range(len(ds["paths"]))]


def max_cumulative_demand(ds: Dict[str, Any], demand_idx: int = 0) -> float:
    worst = 0.0
    for p in ds["paths"]:
        cum = 0.0
        for row in p["values"]:
            cum += max(0.0, float(row[demand_idx]))
            worst = max(worst, cum)
    return worst


def planning_candidates(ds: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> List[float]:
    params = planning_parameters(ds, params)
    if params.get("candidates"):
        return sorted(set(round(float(x), 8) for x in params["candidates"]))

    demand_idx = dimension_index(ds, str(params["demandDimension"]), 0) or 0
    q_min = float(params.get("qMin", 0.0))
    q_max = params.get("qMax")
    if q_max is None:
        q_max = max_cumulative_demand(ds, demand_idx) * 1.02
    q_max = float(q_max)
    step = float(params.get("qStep", 0.0) or 0.0)
    span = q_max - q_min
    max_points = int(params.get("maxCandidateCount", 101) or 101)
    if step <= 0:
        step = max(1.0, span / 90.0)
    if max_points > 1 and span > 0:
        step = max(step, span / (max_points - 1))

    candidates = [q_min]
    q = q_min
    while q <= q_max + 1e-9:
        candidates.append(round(q, 8))
        q += step
    candidates.append(round(q_max, 8))

    if params.get("exactBreakpoints"):
        for p in ds["paths"]:
            cum = 0.0
            for row in p["values"]:
                cum += max(0.0, float(row[demand_idx]))
                if q_min - 1e-9 <= cum <= q_max + 1e-9:
                    candidates.append(round(cum, 8))
    return sorted(set(candidates))


def evaluate_q(
    ds: Dict[str, Any],
    q: float,
    params: Optional[Dict[str, Any]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    params = planning_parameters(ds, params)
    dims = _dim_map(ds, params)
    if weights is None:
        weights = [float(p["probability"]) for p in ds["paths"]]
    total_w = math.fsum(max(0.0, float(w)) for w in weights)
    if total_w <= 0:
        total_w = 1.0

    cost = 0.0
    emergency_units = 0.0
    served = 0.0
    demand = 0.0
    service_weighted = 0.0
    for p, w0 in zip(ds["paths"], weights):
        w = max(0.0, float(w0))
        sim = simulate_path(p["values"], q, params, dims)
        cost += w * sim["totalCost"]
        emergency_units += w * sim["emergencyUnits"]
        served += w * sim["servedUnits"]
        demand += w * sim["totalDemand"]
        service_weighted += w * sim["serviceLevel"]

    return {
        "q": float(q),
        "expectedCost": cost / total_w,
        "expectedEmergencyUnits": emergency_units / total_w,
        "expectedServiceLevel": service_weighted / total_w,
        "fillRate": served / demand if demand > 1e-12 else 1.0,
        "probabilityMass": total_w,
    }


def solve_planning(
    ds: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    candidates: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Exhaustively evaluate every candidate Q and return the best decision."""
    params = planning_parameters(ds, params)
    if candidates is None:
        candidates = planning_candidates(ds, params)
    candidates = sorted(set(float(q) for q in candidates))
    if weights is None:
        weights = [float(p["probability"]) for p in ds["paths"]]
    fast = FastPlanningCost(ds, params)
    curve = [{"q": q, "expectedCost": fast.expected_cost(q, weights)} for q in candidates]
    best_point = min(curve, key=lambda x: x["expectedCost"])
    best_q = best_point["q"]

    # Optional local refinement around the best coarse-grid point.  The caller
    # can pass the same expanded candidate list to both full and reduced runs.
    if params.get("refine") and len(candidates) >= 2:
        ordered = sorted(candidates)
        pos = min(range(len(ordered)), key=lambda i: abs(ordered[i] - best_q))
        lo = ordered[max(0, pos - 1)]
        hi = ordered[min(len(ordered) - 1, pos + 1)]
        step = min(abs(ordered[1] - ordered[0]), hi - lo) / 10.0
        if step > 0:
            extra = []
            x = lo
            while x <= hi + 1e-9:
                extra.append(round(x, 8))
                x += step
            existing = {round(q, 8) for q in candidates}
            extra = [q for q in extra if round(q, 8) not in existing]
            if extra:
                extra_curve = [{"q": q, "expectedCost": fast.expected_cost(q, weights)} for q in extra]
                curve.extend(extra_curve)
                curve.sort(key=lambda x: x["q"])
                best_point = min(curve, key=lambda x: x["expectedCost"])
                best_q = best_point["q"]

    best = evaluate_q(ds, best_q, params, weights)
    losses = [fast.path_cost(i, best_q) for i in range(len(ds["paths"]))]
    risk = risk_summary(losses, list(weights), float(params.get("tailAlpha", 0.95)), ids=[p["id"] for p in ds["paths"]])
    return {
        "decision": best,
        "curve": curve,
        "referenceLosses": losses,
        "riskAtDecision": risk,
        "parameters": params,
        "candidateCount": len(candidates),
        "solver": "exhaustive-grid",
    }


def compare_decisions(
    full_ds: Dict[str, Any],
    reduced_ds: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve both sets with one model and evaluate both Qs on the original set."""
    params = planning_parameters(full_ds, params)
    candidates = planning_candidates(full_ds, params)
    full = solve_planning(full_ds, params, candidates)
    reduced = solve_planning(reduced_ds, params, candidates)

    q_full = full["decision"]["q"]
    q_red = reduced["decision"]["q"]
    red_q_on_full = evaluate_q(full_ds, q_red, params)
    full_q_on_full = evaluate_q(full_ds, q_full, params)
    q_full_on_reduced = evaluate_q(reduced_ds, q_full, params)

    full_losses_at_red_q = path_loss_values(full_ds, q_red, params)
    weights = [float(p["probability"]) for p in full_ds["paths"]]
    alpha = float(params.get("tailAlpha", 0.95))
    risk_red_q_full = risk_summary(full_losses_at_red_q, weights, alpha, ids=[p["id"] for p in full_ds["paths"]])

    regret = red_q_on_full["expectedCost"] - full_q_on_full["expectedCost"]
    return {
        "parameters": params,
        "candidateCount": len(candidates),
        "candidateRange": [min(candidates), max(candidates)],
        "full": full,
        "reduced": reduced,
        "qOriginal": q_full,
        "qReduced": q_red,
        "decisionDelta": q_red - q_full,
        "decisionDeltaPercent": (100.0 * (q_red - q_full) / q_full) if abs(q_full) > 1e-9 else None,
        "originalObjective": full_q_on_full["expectedCost"],
        "reducedObjectiveInSample": reduced["decision"]["expectedCost"],
        "reducedDecisionOnOriginal": red_q_on_full,
        "originalDecisionOnReduced": q_full_on_reduced,
        "outOfSampleRegret": regret,
        "relativeRegretPercent": 100.0 * regret / max(1e-9, abs(full_q_on_full["expectedCost"])),
        "tailRiskOfReducedQOnOriginal": risk_red_q_full,
    }
