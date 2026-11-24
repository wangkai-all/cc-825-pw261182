"""End-to-end analysis: validation, reduction, planning, Pareto and checks."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .jobs import recovery_drill as run_recovery_drill
from .planning import (
    evaluate_q,
    path_loss_values,
    planning_candidates,
    planning_parameters,
    solve_planning,
)
from .reduction import exhaustive_reduce, heuristic_reduce, reduce_scenarios
from .risk import risk_summary
from .sample import toy_dataset
from .dataset import prepare_dataset


def _planning_comparison(
    ds: Dict[str, Any],
    reduced_ds: Dict[str, Any],
    planning_options: Optional[Dict[str, Any]],
    alpha: float,
    full_plan: Optional[Dict[str, Any]] = None,
    candidates: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    params = planning_parameters(ds, planning_options)
    params["tailAlpha"] = alpha
    if candidates is None:
        candidates = planning_candidates(ds, params)
    if full_plan is None:
        full_plan = solve_planning(ds, params, candidates)
    reduced = solve_planning(reduced_ds, params, candidates)
    q_full = full_plan["decision"]["q"]
    q_red = reduced["decision"]["q"]

    red_on_full = evaluate_q(ds, q_red, params)
    full_on_full = full_plan["decision"]
    full_q_on_reduced = evaluate_q(reduced_ds, q_full, params)
    losses_red_q = path_loss_values(ds, q_red, params)
    weights = [float(p["probability"]) for p in ds["paths"]]
    tail_red_q = risk_summary(losses_red_q, weights, alpha, ids=[p["id"] for p in ds["paths"]])
    regret = red_on_full["expectedCost"] - full_on_full["expectedCost"]

    return {
        "parameters": params,
        "candidateCount": len(candidates),
        "candidateRange": [min(candidates), max(candidates)],
        "originalCurve": full_plan["curve"],
        "reducedCurve": reduced["curve"],
        "qOriginal": q_full,
        "qReduced": q_red,
        "decisionDelta": q_red - q_full,
        "decisionDeltaPercent": 100.0 * (q_red - q_full) / q_full if abs(q_full) > 1e-9 else None,
        "originalObjective": full_on_full["expectedCost"],
        "reducedObjectiveInSample": reduced["decision"]["expectedCost"],
        "reducedDecisionOnOriginal": red_on_full,
        "originalDecisionOnReduced": full_q_on_reduced,
        "outOfSampleRegret": regret,
        "relativeRegretPercent": 100.0 * regret / max(1e-9, abs(full_on_full["expectedCost"])),
        "tailRiskOfReducedQOnOriginal": tail_red_q,
        "solver": "same exhaustive candidate planning model before and after reduction",
    }


def run_analysis(payload: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options = dict(options or {})
    ds, validation = prepare_dataset(
        payload,
        normalize=bool(options.get("normalize", False)),
        drop_zero=bool(options.get("dropZero", False)),
        strict=bool(options.get("strict", False)),
    )

    alpha = float(options.get("tailLevel", 0.90))
    planning_options = options.get("planningParameters") or options.get("planning") or {}
    params = planning_parameters(ds, planning_options)
    candidates = planning_candidates(ds, params)
    full_plan = solve_planning(ds, params, candidates)

    reduction_options = dict(options)
    reduction_options["losses"] = full_plan["referenceLosses"]
    reduction_options["planningParameters"] = params
    reduction = reduce_scenarios(ds, reduction_options)

    comparison = _planning_comparison(
        ds,
        reduction["reducedDataset"],
        planning_options,
        alpha,
        full_plan=full_plan,
        candidates=candidates,
    )

    return {
        "datasetName": ds.get("name"),
        "validation": validation,
        "dimensions": ds["dimensions"],
        "periods": ds["periods"],
        "preparedPaths": ds["paths"],
        "fullPlanning": {k: v for k, v in full_plan.items() if k != "referenceLosses"},
        "reduction": reduction,
        "planningComparison": comparison,
    }


def run_pareto(
    payload: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    options = dict(options or {})
    ds, validation = prepare_dataset(
        payload,
        normalize=bool(options.get("normalize", False)),
        drop_zero=bool(options.get("dropZero", False)),
    )
    alpha = float(options.get("tailLevel", 0.90))
    planning_options = options.get("planningParameters") or {}
    params = planning_parameters(ds, planning_options)
    candidates = planning_candidates(ds, params)
    full_plan = solve_planning(ds, params, candidates)

    grid = list(weights or [0.0, 0.25, 0.75, 1.5, 3.0, 5.0])
    points: List[Dict[str, Any]] = []
    for w in grid:
        opts = dict(options)
        opts["tailWeight"] = float(w)
        opts["losses"] = full_plan["referenceLosses"]
        opts["planningParameters"] = params
        red = reduce_scenarios(ds, opts)
        reduced_ds = red["reducedDataset"]
        red_plan = solve_planning(reduced_ds, params, candidates)
        q_red = red_plan["decision"]["q"]
        eval_full = evaluate_q(ds, q_red, params)
        regret = eval_full["expectedCost"] - full_plan["decision"]["expectedCost"]
        m = red["metrics"]
        points.append({
            "tailWeight": float(w),
            "method": red["method"],
            "objective": red["objective"],
            "averageDistance": m["averageDistance"],
            "maxAssignedDistance": m["maxAssignedDistance"],
            "cvarOriginal": m["originalRisk"]["cvar"],
            "cvarReduced": m["reducedRisk"]["cvar"],
            "cvarError": m["cvarError"],
            "cvarUnderestimate": m["cvarUnderestimate"],
            "absoluteCvarError": m["absoluteCvarError"],
            "worstOriginal": m["originalRisk"]["worst"],
            "worstReduced": m["reducedRisk"]["worst"],
            "worstUnderestimate": m["worstUnderestimate"],
            "tailRoutingWithinTolerance": m["tailRoutingWithinTolerance"],
            "tailRoutingToTailRepresentatives": m["tailRoutingToTailRepresentatives"],
            "qReduced": q_red,
            "planningRegret": regret,
            "relativeRegretPercent": 100.0 * regret / max(1e-9, abs(full_plan["decision"]["expectedCost"])),
            "probabilityError": m["probabilityConservation"]["error"],
            "representativeIds": [r["id"] for r in m["representatives"]],
        })

    min_distance = min(p["averageDistance"] for p in points)
    min_abs_cvar = min(p["absoluteCvarError"] for p in points)
    min_regret = min(p["planningRegret"] for p in points)
    for p in points:
        p["distancePriceOfTailProtection"] = p["averageDistance"] - min_distance
        p["tailFidelityCost"] = p["absoluteCvarError"] - min_abs_cvar
        p["planningRegretCost"] = p["planningRegret"] - min_regret

    distance_first = min(points, key=lambda p: p["tailWeight"])
    tail_strong = max(points, key=lambda p: p["tailWeight"])
    return {
        "validationSummary": {
            "pathCountAfterDuplicateMerge": validation["pathCountAfterDuplicateMerge"],
            "duplicateRowsMerged": validation["duplicateRowsMerged"],
            "probability": validation["probabilities"],
        },
        "grid": points,
        "utopia": {
            "minimumAverageDistance": min_distance,
            "minimumAbsoluteCvarError": min_abs_cvar,
            "minimumPlanningRegret": min_regret,
        },
        "costs": {
            "distanceFirstDistance": distance_first["averageDistance"],
            "distanceFirstCvarUnderestimate": distance_first["cvarUnderestimate"],
            "distanceFirstPlanningRegret": distance_first["planningRegret"],
            "tailStrongDistancePremium": tail_strong["distancePriceOfTailProtection"],
            "tailStrongResidualCvarError": tail_strong["absoluteCvarError"],
        },
    }


def run_exact_check(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options = options or {}
    payload = options.get("dataset") or toy_dataset()
    ds, validation = prepare_dataset(payload, normalize=bool(options.get("normalize", True)))
    k = int(options.get("k", 3))
    base = {
        "k": k,
        "tailWeight": float(options.get("tailWeight", 0.75)),
        "tailLevel": float(options.get("tailLevel", 0.90)),
        "tailGuards": 0,  # complete search over every C(N,K) subset
        "prefixLocked": True,
        "method": "exhaustive",
        "seed": 42,
    }
    params = planning_parameters(ds, options.get("planningParameters") or {})
    losses = path_loss_values(ds, solve_planning(ds, params, planning_candidates(ds, params))["decision"]["q"], params)
    base["losses"] = losses

    exact = exhaustive_reduce(ds, base)
    heuristic = heuristic_reduce(ds, {**base, "method": "heuristic-pam"})
    exact_objective = exact["objective"]
    heur_objective = heuristic["objective"]
    gap = heur_objective - exact_objective
    relative_gap = gap / max(abs(exact_objective), 1e-9)

    return {
        "instance": {
            "name": ds["name"],
            "pathCount": len(ds["paths"]),
            "k": k,
            "subsets": math.comb(len(ds["paths"]), k),
        },
        "validation": validation,
        "exact": _compact_check_result(exact),
        "heuristic": _compact_check_result(heuristic),
        "optimality": {
            "exactObjective": exact_objective,
            "heuristicObjective": heur_objective,
            "absoluteGap": gap,
            "relativeGap": relative_gap,
            "matchesExhaustiveOptimum": abs(gap) <= 1e-7 * max(1.0, abs(exact_objective)),
        },
    }


def _compact_check_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": result["method"],
        "objective": result["objective"],
        "representativeIds": [r["id"] for r in result["representatives"]],
        "representativeIndices": result["representativeIndices"],
        "probabilityConservation": result["probabilityConservation"],
        "nonAnticipativity": {
            "compatible": result["nonAnticipativity"].get("compatible"),
            "violationCount": len(result["nonAnticipativity"].get("violations", [])),
            "rule": result["nonAnticipativity"].get("rule"),
        },
        "treeValidation": result["treeValidation"],
        "search": result.get("search", {}),
    }


def run_recovery_check() -> Dict[str, Any]:
    return run_recovery_drill()
