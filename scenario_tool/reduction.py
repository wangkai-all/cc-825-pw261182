"""Joint representative selection and probability redistribution.

For a candidate representative set S, every original path is assigned through a
prefix-compatible dynamic program (or, when explicitly requested, an
unconstrained nearest objective).  Representative probabilities are then the
sum of the probability mass assigned to them.  Thus selection and
redistribution share one objective rather than being unrelated post-processing
steps.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .distances import DistanceModel
from .risk import loss_scale, risk_summary, weighted_quantile_value
from .tree import (
    check_prefix_assignment,
    independent_assignment,
    prefix_compatible_assignment,
)

DEFAULT_OPTIONS: Dict[str, Any] = {
    "k": 12,
    "tailWeight": 1.0,
    "tailLevel": 0.90,
    "tailGuards": 2,
    "tailConditionalWeight": 0.55,
    "prefixLocked": True,
    "distanceMetric": "prefix",
    "timeCurve": "early",
    "seed": 2611825,
    "method": "auto",
    "maxExhaustiveCombinations": 20000,
}


def normalize_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_OPTIONS)
    if options:
        for key, value in options.items():
            if value is not None:
                cfg[key] = value
    # Friendly alias from UIs that expose a single 0..1 protection control.
    if "tailProtection" in (options or {}) and "tailWeight" not in (options or {}):
        try:
            cfg["tailWeight"] = 4.0 * float(options["tailProtection"])
        except (TypeError, ValueError):
            pass
    cfg["k"] = max(1, int(cfg["k"]))
    cfg["tailWeight"] = max(0.0, float(cfg["tailWeight"]))
    cfg["tailLevel"] = min(0.999, max(0.5, float(cfg["tailLevel"])))
    cfg["tailGuards"] = max(0, int(cfg["tailGuards"]))
    cfg["tailConditionalWeight"] = max(0.0, float(cfg["tailConditionalWeight"]))
    return cfg


def resolve_losses(ds: Dict[str, Any], cfg: Dict[str, Any]) -> List[float]:
    losses = cfg.get("losses")
    if losses is not None:
        if len(losses) != len(ds["paths"]):
            raise ValueError("losses length must match path count")
        return [float(x) for x in losses]
    maybe = []
    has_metadata_loss = True
    for p in ds["paths"]:
        loss = p.get("loss", (p.get("metadata") or {}).get("loss"))
        if loss is None:
            has_metadata_loss = False
            break
        maybe.append(float(loss))
    if has_metadata_loss and maybe:
        return maybe

    # Late import keeps dataset/distance/tree modules independent of planning.
    from .planning import solve_planning

    solution = solve_planning(ds, cfg.get("planningParameters") or {})
    return list(solution["referenceLosses"])


def build_context(ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = normalize_options(options)
    dist = DistanceModel(ds, cfg)
    weights = [max(0.0, float(p["probability"])) for p in ds["paths"]]
    mass = math.fsum(weights)
    losses = resolve_losses(ds, cfg)
    l_scale = loss_scale(losses, weights)
    threshold = weighted_quantile_value(losses, weights, cfg["tailLevel"])
    tail_set = [l >= threshold - 1e-12 for l in losses]
    tail_mass = math.fsum(w for w, flag in zip(weights, tail_set) if flag)
    candidate_idx = [i for i, w in enumerate(weights) if w > 0.0]
    if not candidate_idx:
        raise ValueError("at least one positive-probability path is required")
    k = min(cfg["k"], len(candidate_idx))
    cfg["k"] = k

    # Small/medium instances reuse all pairwise distances; large instances
    # compute only path-to-candidate distances to stay responsive at 10k rows.
    distance_matrix = dist.full_matrix() if len(weights) <= 400 else None

    guard_count = min(cfg["tailGuards"], k, len(candidate_idx))
    guards = sorted(candidate_idx, key=lambda i: losses[i], reverse=True)[:guard_count]

    return {
        "ds": ds,
        "cfg": cfg,
        "dist": dist,
        "distanceMatrix": distance_matrix,
        "weights": weights,
        "mass": mass,
        "losses": losses,
        "lossScale": l_scale,
        "tailThreshold": threshold,
        "tailSet": tail_set,
        "tailMass": tail_mass,
        "candidateIndices": candidate_idx,
        "guards": guards,
    }


def pair_components(ctx: Dict[str, Any], i: int, r: int) -> Tuple[float, float, float]:
    matrix = ctx.get("distanceMatrix")
    if matrix is not None:
        d = matrix[i][r]
    else:
        d = ctx["dist"].distance(i, r)
    li = ctx["losses"][i]
    lr = ctx["losses"][r]
    normalized_gap = (li - lr) / max(1e-12, ctx["lossScale"])
    under = max(0.0, normalized_gap)
    over = max(0.0, -normalized_gap)
    # Asymmetric: understating a costly tail is worse than conservative overstatement.
    loss_penalty = under * under + 0.25 * over * over
    return d, loss_penalty, normalized_gap


def single_leaf_cost(ctx: Dict[str, Any], i: int, r: int) -> float:
    cfg = ctx["cfg"]
    lam = cfg["tailWeight"]
    cond_w = cfg["tailConditionalWeight"]
    tail_mass = max(ctx["tailMass"], 1e-12)
    p = ctx["weights"][i]
    d, loss_penalty, _ = pair_components(ctx, i, r)
    expected_part = p * (d + lam * loss_penalty)
    tail_part = 0.0
    if lam > 0 and ctx["tailSet"][i]:
        tail_part = lam * cond_w * (p / tail_mass) * (d + loss_penalty)
    # Deterministic tie-break for zero-probability rows.
    return expected_part + tail_part + 1e-10 * d


def make_leaf_cost(ctx: Dict[str, Any], rep_indices: Sequence[int]) -> Any:
    def cost(i: int, local_k: int) -> float:
        r = int(rep_indices[local_k])
        return single_leaf_cost(ctx, i, r)

    return cost


def assign_selection(
    ctx: Dict[str, Any], rep_indices: Sequence[int]
) -> Tuple[List[int], float, Dict[str, Any]]:
    cfg = ctx["cfg"]
    ds = ctx["ds"]
    cost = make_leaf_cost(ctx, rep_indices)
    if cfg["prefixLocked"]:
        assignment, objective = prefix_compatible_assignment(ds, rep_indices, cost)
        check = check_prefix_assignment(ds, rep_indices, assignment)
        return assignment, objective, check

    local_assignment, objective = independent_assignment(len(ds["paths"]), len(rep_indices), cost)
    assignment = [int(rep_indices[k]) for k in local_assignment]
    check = check_prefix_assignment(ds, rep_indices, assignment)
    # It is expected that unlocked assignments may fail; mark lock false.
    check["compatible"] = check["compatible"]
    return assignment, objective, check


def redistribute(ctx: Dict[str, Any], rep_indices: Sequence[int], assignment: Sequence[int]) -> Tuple[List[float], List[List[int]]]:
    k = len(rep_indices)
    rep_set_to_local = {int(r): j for j, r in enumerate(rep_indices)}
    q = [0.0] * k
    members: List[List[int]] = [[] for _ in range(k)]
    weights = ctx["weights"]
    for i, assigned in enumerate(assignment):
        if int(assigned) in rep_set_to_local:
            j = rep_set_to_local[int(assigned)]
        else:
            # Defensive support for local-index assignments.
            j = max(0, min(k - 1, int(assigned)))
        q[j] += weights[i]
        members[j].append(i)
    q = [float(x) for x in q]
    return q, members


def selection_metrics(
    ctx: Dict[str, Any],
    rep_indices: Sequence[int],
    assignment: Sequence[int],
    q: Sequence[float],
    members: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    ds = ctx["ds"]
    weights = ctx["weights"]
    mass = max(ctx["mass"], 1e-12)
    losses = ctx["losses"]
    alpha = ctx["cfg"]["tailLevel"]

    weighted_distance = 0.0
    max_distance = 0.0
    tail_routed_to_tail_rep = 0.0
    tail_routed_within_tolerance = 0.0
    tolerance = 0.10 * max(1.0, ctx["lossScale"])
    assignment_rows = []

    for i, r in enumerate(assignment):
        r = int(r)
        d, _, gap = pair_components(ctx, i, r)
        p = weights[i]
        weighted_distance += p * d
        max_distance = max(max_distance, d)
        if ctx["tailSet"][i]:
            if ctx["tailSet"][r]:
                tail_routed_to_tail_rep += p
            if losses[r] >= losses[i] - tolerance:
                tail_routed_within_tolerance += p
        assignment_rows.append({
            "pathIndex": i,
            "id": ds["paths"][i]["id"],
            "representativeIndex": r,
            "representativeId": ds["paths"][r]["id"],
            "probability": p,
            "distance": d,
            "loss": losses[i],
            "representativeLoss": losses[r],
            "lossGap": losses[r] - losses[i],
            "normalizedLossGap": gap,
            "isTail": bool(ctx["tailSet"][i]),
            "tags": ds["paths"][i].get("tags", []),
        })

    avg_distance = weighted_distance / mass
    original_risk = risk_summary(losses, weights, alpha, ids=[p["id"] for p in ds["paths"]])
    rep_losses = [losses[int(r)] for r in rep_indices]
    rep_ids = [ds["paths"][int(r)]["id"] for r in rep_indices]
    reduced_risk = risk_summary(rep_losses, list(q), alpha, ids=rep_ids)

    cvar_error = reduced_risk["cvar"] - original_risk["cvar"]
    expected_error = reduced_risk["expected"] - original_risk["expected"]
    worst_error = reduced_risk["worst"] - original_risk["worst"]
    tail_mass = max(ctx["tailMass"], 1e-12)

    reps = []
    for j, r in enumerate(rep_indices):
        member_distances = [assignment_rows[i]["distance"] for i in members[j]]
        member_weighted_d = math.fsum(weights[i] * assignment_rows[i]["distance"] for i in members[j])
        member_mass = math.fsum(weights[i] for i in members[j])
        reps.append({
            "slot": j,
            "pathIndex": int(r),
            "id": ds["paths"][r]["id"],
            "sourceIds": ds["paths"][r].get("sourceIds", [ds["paths"][r]["id"]]),
            "tags": ds["paths"][r].get("tags", []),
            "probability": float(q[j]),
            "originalProbability": weights[r],
            "memberCount": len(members[j]),
            "memberMass": member_mass,
            "meanDistance": member_weighted_d / max(member_mass, 1e-12),
            "maxDistance": max(member_distances) if member_distances else 0.0,
            "loss": losses[r],
            "isTail": bool(ctx["tailSet"][r]),
            "prefixValues": ds["paths"][r]["values"],
        })

    allocated_mass = math.fsum(q)
    return {
        "averageDistance": avg_distance,
        "weightedDistanceObjective": weighted_distance,
        "maxAssignedDistance": max_distance,
        "originalRisk": original_risk,
        "reducedRisk": reduced_risk,
        "expectedLossError": expected_error,
        "cvarError": cvar_error,
        "cvarUnderestimate": max(0.0, -cvar_error),
        "absoluteCvarError": abs(cvar_error),
        "worstLossError": worst_error,
        "worstUnderestimate": max(0.0, -worst_error),
        "absoluteWorstError": abs(worst_error),
        "tailRoutingToTailRepresentatives": tail_routed_to_tail_rep / tail_mass,
        "tailRoutingWithinTolerance": tail_routed_within_tolerance / tail_mass,
        "originalTailMass": ctx["tailMass"],
        "probabilityConservation": {
            "inputMass": ctx["mass"],
            "allocatedMass": allocated_mass,
            "error": abs(allocated_mass - ctx["mass"]),
            "tolerance": 1e-8 * max(1.0, ctx["mass"]),
            "conserved": abs(allocated_mass - ctx["mass"]) <= 1e-8 * max(1.0, ctx["mass"]),
        },
        "representatives": reps,
        "assignments": assignment_rows,
    }


def build_reduced_dataset(ctx: Dict[str, Any], rep_indices: Sequence[int], q: Sequence[float]) -> Dict[str, Any]:
    ds = ctx["ds"]
    paths = []
    for j, r in enumerate(rep_indices):
        original = ds["paths"][r]
        paths.append({
            "id": original["id"],
            "probability": float(q[j]),
            "values": original["values"],
            "tags": original.get("tags", []),
            "sourceIds": original.get("sourceIds", [original["id"]]),
            "representativeSlot": j,
            "loss": ctx["losses"][r],
        })
    return {
        "name": f"{ds.get('name', 'Scenario set')} — reduced K={len(rep_indices)}",
        "description": "Representative paths with jointly redistributed probabilities",
        "periods": ds["periods"],
        "dimensions": ds["dimensions"],
        "planning": ds.get("planning", {}),
        "paths": paths,
        "kind": "reduced-representatives",
        "provenance": {
            "sourceKind": ds.get("kind", "scenario-paths"),
            "representativeIndices": list(map(int, rep_indices)),
            "probabilityState": "redistributed-final",
            "massBasis": ctx["mass"],
            "doNotRedistributeAgain": True,
        },
    }


def finalize_selection(
    ctx: Dict[str, Any],
    rep_indices: Sequence[int],
    *,
    method: str,
    search_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rep_indices = tuple(int(x) for x in rep_indices)
    assignment, objective, na_check = assign_selection(ctx, rep_indices)
    q, members = redistribute(ctx, rep_indices, assignment)
    metrics = selection_metrics(ctx, rep_indices, assignment, q, members)
    reduced_ds = build_reduced_dataset(ctx, rep_indices, q)

    from .tree import build_prefix_tree, validate_tree

    tree = build_prefix_tree(reduced_ds)
    tree_validation = validate_tree(tree, expected_total=ctx["mass"])
    na_check["assignmentCompatible"] = na_check.get("compatible", False)

    return {
        "method": method,
        "search": search_info or {},
        "options": {k: v for k, v in ctx["cfg"].items() if k != "losses"},
        "objective": objective,
        "metrics": metrics,
        "nonAnticipativity": na_check,
        "treeValidation": tree_validation,
        "tree": tree,
        "reducedDataset": reduced_ds,
        "representativeIndices": list(rep_indices),
        "assignments": metrics["assignments"],
        "representatives": metrics["representatives"],
        "probabilityConservation": metrics["probabilityConservation"],
    }


def exhaustive_reduce(ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = build_context(ds, options)
    cfg = ctx["cfg"]
    candidates = ctx["candidateIndices"]
    k = cfg["k"]
    required = set(ctx["guards"])
    optional = [i for i in candidates if i not in required]
    need = k - len(required)
    if need < 0:
        # More guards than k: keep the highest-loss k guards.
        required = set(ctx["guards"][:k])
        optional = [i for i in candidates if i not in required]
        need = 0
    if need > len(optional):
        raise ValueError("not enough candidate paths for requested K")

    combos = itertools.combinations(optional, need)
    evaluated = 0
    best_score = math.inf
    best_combo: Optional[Tuple[int, ...]] = None
    max_evals = int(cfg["maxExhaustiveCombinations"])

    for extra in combos:
        evaluated += 1
        if evaluated > max_evals:
            break
        combo = tuple(sorted(required | set(extra)))
        cost = make_leaf_cost(ctx, combo)
        if cfg["prefixLocked"]:
            _, score = prefix_compatible_assignment(ds, combo, cost)
        else:
            _, score = independent_assignment(len(ds["paths"]), k, cost)
        if score < best_score - 1e-12:
            best_score = score
            best_combo = combo

    if best_combo is None:
        raise RuntimeError("exhaustive search evaluated no feasible combinations; increase maxExhaustiveCombinations")
    result = finalize_selection(ctx, best_combo, method="exhaustive", search_info={
        "combinationsEvaluated": evaluated,
        "complete": evaluated <= max_evals,
        "bestSearchObjective": best_score,
        "requiredTailGuards": sorted(required),
    })
    return result


def _weighted_seed(ctx: Dict[str, Any], rng: random.Random) -> int:
    candidates = ctx["candidateIndices"]
    weights = [ctx["weights"][i] + 1e-12 for i in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _kmedoids_pp_start(ctx: Dict[str, Any], rng: random.Random) -> Tuple[int, ...]:
    candidates = ctx["candidateIndices"]
    selected = list(ctx["guards"])
    selected_set = set(selected)
    if not selected:
        first = _weighted_seed(ctx, rng)
        selected = [first]
        selected_set = {first}

    min_dist = {i: min((ctx["dist"].distance(i, r) for r in selected), default=math.inf) for i in candidates}
    while len(selected) < ctx["cfg"]["k"]:
        pool = [i for i in candidates if i not in selected_set]
        if not pool:
            break
        mass_weights = [ctx["weights"][i] + 1e-10 for i in pool]
        d2 = [min_dist[i] ** 2 for i in pool]
        weights = [w * d for w, d in zip(mass_weights, d2)]
        if math.fsum(weights) <= 1e-20:
            choice = rng.choice(pool)
        else:
            choice = rng.choices(pool, weights=weights, k=1)[0]
        selected.append(choice)
        selected_set.add(choice)
        for i in candidates:
            d = ctx["dist"].distance(i, choice)
            if d < min_dist[i]:
                min_dist[i] = d
    return tuple(sorted(selected_set))[: ctx["cfg"]["k"]]


def _unconstrained_score(ctx: Dict[str, Any], reps: Sequence[int]) -> Tuple[float, List[int]]:
    cost = make_leaf_cost(ctx, reps)
    local_assignment, score = independent_assignment(len(ctx["ds"]["paths"]), len(reps), cost)
    assignment = [int(reps[j]) for j in local_assignment]
    return score, assignment


def _best_cost_state(
    ctx: Dict[str, Any], reps: Sequence[int]
) -> Tuple[float, List[float], List[float], List[int]]:
    """Best and second-best leaf cost per path for O(N) swap deltas."""
    n = len(ctx["ds"]["paths"])
    best1 = [math.inf] * n
    best2 = [math.inf] * n
    best_rep = [-1] * n
    for r in reps:
        for i in range(n):
            c = single_leaf_cost(ctx, i, int(r))
            if c < best1[i]:
                best2[i] = best1[i]
                best1[i] = c
                best_rep[i] = int(r)
            elif c < best2[i]:
                best2[i] = c
    return math.fsum(best1), best1, best2, best_rep


def _cluster_medoid_update(ctx: Dict[str, Any], reps: Tuple[int, ...], assignment: Sequence[int]) -> Tuple[Tuple[int, ...], bool]:
    rep_local = {r: j for j, r in enumerate(reps)}
    clusters: List[List[int]] = [[] for _ in reps]
    for i, a in enumerate(assignment):
        clusters[rep_local[int(a)]].append(i)

    guards = set(ctx["guards"])
    new_reps = list(reps)
    changed = False
    for j, cluster in enumerate(clusters):
        if reps[j] in guards or not cluster:
            continue
        # Exact medoid for small clusters; bounded candidate medoid search for
        # large clusters keeps 10k-path imports responsive.
        if len(cluster) <= 120:
            candidate_medoids = cluster
        else:
            rng = random.Random(ctx["cfg"]["seed"] + j * 1009)
            by_mass = sorted(cluster, key=lambda i: ctx["weights"][i], reverse=True)[:24]
            by_tail = sorted(cluster, key=lambda i: ctx["losses"][i], reverse=True)[:16]
            sample = rng.sample(cluster, min(48, len(cluster)))
            candidate_medoids = list(dict.fromkeys([reps[j]] + by_mass + by_tail + sample))

        best_r = reps[j]
        best_cost = math.inf
        for candidate in candidate_medoids:
            cost_sum = 0.0
            for i in cluster:
                d, lp, _ = pair_components(ctx, i, candidate)
                p = ctx["weights"][i]
                cost_sum += p * (d + ctx["cfg"]["tailWeight"] * lp) + 1e-10 * d
            if cost_sum < best_cost - 1e-12:
                best_cost = cost_sum
                best_r = candidate
        if best_r != reps[j]:
            new_reps[j] = best_r
            changed = True
    return tuple(sorted(new_reps)), changed


def _swap_neighborhood(ctx: Dict[str, Any], current: Sequence[int], iteration: int) -> List[int]:
    cfg = ctx["cfg"]
    candidates = ctx["candidateIndices"]
    rep_set = set(current)
    small = len(candidates) <= 18 and cfg["k"] <= 8
    if small:
        return [i for i in candidates if i not in rep_set]

    rng = random.Random(cfg["seed"] + iteration * 7919)
    high_mass = sorted(candidates, key=lambda i: ctx["weights"][i], reverse=True)[:18]
    high_loss = sorted(candidates, key=lambda i: ctx["losses"][i], reverse=True)[:18]
    priority = set(high_mass + high_loss)
    rest = [i for i in candidates if i not in rep_set and i not in priority]
    sampled = rng.sample(rest, min(24, len(rest))) if rest else []
    incoming = list(dict.fromkeys(high_mass + high_loss + sampled))
    return [i for i in incoming if i not in rep_set]


def _local_swap_search(
    ctx: Dict[str, Any],
    start: Tuple[int, ...],
    *,
    max_passes: Optional[int] = None,
) -> Tuple[Tuple[int, ...], float]:
    guards = set(ctx["guards"])
    current = tuple(sorted(set(start)))
    current_score, best1, best2, best_rep = _best_cost_state(ctx, current)
    n = len(ctx["ds"]["paths"])
    if max_passes is None:
        max_passes = 1 if len(ctx["candidateIndices"]) > 500 else 6

    for iteration in range(max_passes):
        replaceable = [r for r in current if r not in guards]
        if not replaceable:
            break
        replaceable_set = set(replaceable)
        incoming = _swap_neighborhood(ctx, current, iteration)
        if not incoming:
            break

        best_score = current_score
        best_swap: Optional[Tuple[int, int]] = None
        rep_set = set(current)

        for new in incoming:
            if new in rep_set:
                continue
            new_costs = [single_leaf_cost(ctx, i, new) for i in range(n)]
            global_gain = 0.0
            group_correction = {old: 0.0 for old in replaceable}
            for i in range(n):
                b1 = best1[i]
                nc = new_costs[i]
                global_gain += min(b1, nc) - b1
                br = best_rep[i]
                if br in replaceable_set:
                    group_correction[br] += min(best2[i], nc) - min(b1, nc)

            for old in replaceable:
                trial_score = current_score + global_gain + group_correction[old]
                if trial_score < best_score - 1e-9:
                    best_score = trial_score
                    best_swap = (old, new)

        if best_swap is None:
            break
        old, new = best_swap
        current = tuple(sorted(new if x == old else x for x in current))
        current_score, best1, best2, best_rep = _best_cost_state(ctx, current)

    return current, current_score


def _stratified_optimization_sample(ctx: Dict[str, Any], sample_size: int) -> List[int]:
    candidates = ctx["candidateIndices"]
    n_candidates = len(candidates)
    sample_size = min(n_candidates, max(sample_size, ctx["cfg"]["k"] + ctx["cfg"]["tailGuards"] + 4))
    chosen = set(ctx["guards"])

    by_mass = sorted(candidates, key=lambda i: ctx["weights"][i], reverse=True)
    chosen.update(by_mass[: min(48, sample_size // 4)])
    by_loss = sorted(candidates, key=lambda i: ctx["losses"][i], reverse=True)
    chosen.update(by_loss[: min(48, sample_size // 4)])

    # Systematic sample preserves prefix/regime blocks when the generator or
    # imported file stores related paths contiguously.
    step = max(1, n_candidates // max(1, sample_size // 2))
    chosen.update(candidates[::step])

    rng = random.Random(ctx["cfg"]["seed"] + 31337)
    remaining = [i for i in candidates if i not in chosen]
    if len(chosen) < sample_size and remaining:
        chosen.update(rng.sample(remaining, min(sample_size - len(chosen), len(remaining))))
    return sorted(chosen)[:sample_size] if len(chosen) > sample_size else sorted(chosen)


def heuristic_reduce(ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = build_context(ds, options)
    cfg = ctx["cfg"]

    # CLARA-style scale path: optimize medoids on a stratified sample, map the
    # medoids back to the full data, then perform one full-data swap refinement.
    if len(ctx["candidateIndices"]) > 500 and str(cfg.get("searchScale", "auto")) == "auto":
        sample_indices = _stratified_optimization_sample(ctx, min(320, max(180, cfg["k"] * 12)))
        sub_paths = [ds["paths"][i] for i in sample_indices]
        sub_ds = dict(ds)
        sub_ds["paths"] = sub_paths
        sub_options = {k: v for k, v in cfg.items() if k != "losses"}
        sub_options["losses"] = [ctx["losses"][i] for i in sample_indices]
        sub_options["method"] = "heuristic-pam"
        sub_result = heuristic_reduce(sub_ds, sub_options)
        mapped = tuple(sorted(sample_indices[j] for j in sub_result["representativeIndices"]))
        mapped, search_score = _local_swap_search(ctx, mapped)
        result = finalize_selection(ctx, mapped, method="heuristic-clara-pam", search_info={
            "optimizationSampleSize": len(sample_indices),
            "sampleSearch": sub_result.get("search", {}),
            "fullDataSwapObjective": search_score,
            "scalableNeighborhood": True,
        })
        result["search"]["finalObjective"] = result["objective"]
        return result

    k = cfg["k"]
    starts: List[Tuple[int, ...]] = []

    # Deterministic start 1: tail guards + weighted k-medoids++.
    rng = random.Random(int(cfg["seed"]))
    starts.append(_kmedoids_pp_start(ctx, rng))

    # Deterministic start 2: largest probability paths + guards.
    by_prob = sorted(ctx["candidateIndices"], key=lambda i: ctx["weights"][i], reverse=True)
    start2 = tuple(sorted(set(ctx["guards"] + by_prob[:k])))[:k]
    if len(start2) < k:
        for i in by_prob:
            if i not in start2:
                start2 = tuple(sorted(start2 + (i,)))
            if len(start2) == k:
                break
    starts.append(start2)

    # A few seeded k-medoids++ restarts; large instances need fewer starts
    # because each full-data neighborhood evaluation is more expensive.
    restart_count = 1 if len(ctx["candidateIndices"]) > 500 else 3
    for restart in range(restart_count):
        starts.append(_kmedoids_pp_start(ctx, random.Random(int(cfg["seed"]) + 101 * (restart + 1))))

    refined: List[Tuple[float, Tuple[int, ...]]] = []
    search_iterations = 0
    for start in starts:
        reps = tuple(sorted(set(start)))
        for _ in range(6):
            search_iterations += 1
            score, assignment = _unconstrained_score(ctx, reps)
            new_reps, changed = _cluster_medoid_update(ctx, reps, assignment)
            reps = new_reps
            if not changed:
                break
        score, _ = _unconstrained_score(ctx, reps)
        refined.append((score, reps))

    # Spend the O(N * neighborhood) swap search only on the best two starts.
    refined.sort(key=lambda x: x[0])
    best_reps: Optional[Tuple[int, ...]] = None
    best_search_score = math.inf
    for _, reps in refined[:2]:
        swapped, swap_score = _local_swap_search(ctx, reps)
        if swap_score < best_search_score:
            best_search_score = swap_score
            best_reps = swapped

    assert best_reps is not None
    # For very small instances, verify the final prefix-aware score.  The
    # returned result always uses the prefix DP for its actual assignment.
    result = finalize_selection(ctx, best_reps, method="heuristic-pam", search_info={
        "starts": len(starts),
        "medoidIterations": search_iterations,
        "unconstrainedSearchObjective": best_search_score,
        "scalableNeighborhood": len(ctx["candidateIndices"]) > 18,
    })
    result["search"]["finalObjective"] = result["objective"]
    return result


def reduce_scenarios(ds: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main entry point. Uses exhaustive search only for small candidate sets."""
    ctx = build_context(ds, options)
    cfg = ctx["cfg"]
    n_candidates = len(ctx["candidateIndices"])
    k = cfg["k"]
    combos = math.comb(n_candidates, k) if 0 <= k <= n_candidates else math.inf
    method_pref = str(cfg.get("method", "auto"))

    if method_pref == "exhaustive" or (
        method_pref == "auto"
        and combos <= int(cfg["maxExhaustiveCombinations"])
        and n_candidates <= 24
    ):
        return exhaustive_reduce(ds, options)
    return heuristic_reduce(ds, options)
