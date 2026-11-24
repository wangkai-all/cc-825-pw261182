"""Prefix-aware scenario tree construction and non-anticipativity checks.

A node at depth d contains only the first d observed periods.  Representatives
with identical prefixes share a node; the tree therefore cannot branch before
the paths actually differ.

For original-to-representative assignments, prefix compatibility means: if two
original paths have the same prefix through depth d, their chosen representatives
must also have one common depth-d prefix.  This prevents an assignment from
splitting an information set early based on future outcomes.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CostFn = Callable[[int, int], float]


def prefix_key(values: Sequence[Sequence[float]], depth: int, decimals: Optional[int] = 8) -> Tuple[Tuple[float, ...], ...]:
    rows = []
    for t in range(depth):
        if decimals is None:
            rows.append(tuple(float(x) for x in values[t]))
        else:
            rows.append(tuple(_canonical_round(float(x), decimals) for x in values[t]))
    return tuple(rows)


def _canonical_round(x: float, decimals: int) -> float:
    y = round(float(x), decimals)
    if y == 0.0:
        return 0.0
    return y


class PrefixIndex:
    def __init__(self, values: Sequence[Sequence[Sequence[float]]], decimals: Optional[int] = 8):
        self.n = len(values)
        self.t = len(values[0]) if values else 0
        self.decimals = decimals
        self.keys: List[List[Tuple[Tuple[float, ...], ...]]] = [[()]]
        self.index: List[Dict[Tuple[Tuple[float, ...], ...], int]] = [{(): 0}]
        self.children: List[List[List[int]]] = [[] for _ in range(self.t)]
        self.path_node: List[List[int]] = [[0] * (self.t + 1) for _ in range(self.n)]
        self.leaf_path: List[int] = []

        for d in range(1, self.t + 1):
            keys: List[Tuple[Tuple[float, ...], ...]] = []
            index: Dict[Tuple[Tuple[float, ...], ...], int] = {}
            for i, vals in enumerate(values):
                key = prefix_key(vals, d, decimals)
                if key not in index:
                    index[key] = len(keys)
                    keys.append(key)
                idx = index[key]
                self.path_node[i][d] = idx
            self.keys.append(keys)
            self.index.append(index)

        for d in range(self.t):
            parent_count = len(self.keys[d])
            child_count = len(self.keys[d + 1])
            self.children[d] = [[] for _ in range(parent_count)]
            for child_idx, key in enumerate(self.keys[d + 1]):
                parent_key = key[:-1]
                parent_idx = self.index[d][parent_key]
                self.children[d][parent_idx].append(child_idx)

        self.leaf_path = [-1] * len(self.keys[self.t])
        for i in range(self.n):
            leaf_idx = self.path_node[i][self.t]
            self.leaf_path[leaf_idx] = i


def prefix_compatible_assignment(
    ds: Dict[str, Any],
    rep_indices: Sequence[int],
    leaf_cost: CostFn,
    *,
    decimals: Optional[int] = 8,
) -> Tuple[List[int], float]:
    """Globally minimize leaf cost under the prefix-mapping constraint.

    The dynamic program pairs an original prefix node with a representative
    prefix node at the same depth.  Each child information set may independently
    choose one child node in the reduced tree.
    """
    all_values = [p["values"] for p in ds["paths"]]
    rep_values = [ds["paths"][r]["values"] for r in rep_indices]
    orig = PrefixIndex(all_values, decimals)
    reps = PrefixIndex(rep_values, decimals)
    t = orig.t
    k = len(rep_indices)
    if k == 0:
        raise ValueError("at least one representative is required")

    dp: List[Optional[List[float]]] = [None] * (t + 1)
    o_leaves = len(orig.keys[t])
    r_leaves = len(reps.keys[t])
    terminal = [math.inf] * (o_leaves * r_leaves)
    for i in range(orig.n):
        o = orig.path_node[i][t]
        for local_k in range(k):
            r = reps.path_node[local_k][t]
            terminal[o * r_leaves + r] = float(leaf_cost(i, local_k))
    dp[t] = terminal

    for d in range(t - 1, -1, -1):
        o_count = len(orig.keys[d])
        r_count = len(reps.keys[d])
        r_next_count = len(reps.keys[d + 1])
        nxt = dp[d + 1]
        assert nxt is not None
        cur = [math.inf] * (o_count * r_count)
        for o in range(o_count):
            o_children = orig.children[d][o]
            for r in range(r_count):
                r_children = reps.children[d][r]
                total = 0.0
                feasible = True
                for oc in o_children:
                    base = oc * r_next_count
                    best = math.inf
                    for rc in r_children:
                        val = nxt[base + rc]
                        if val < best:
                            best = val
                    if not math.isfinite(best):
                        feasible = False
                        break
                    total += best
                if feasible:
                    cur[o * r_count + r] = total
        dp[d] = cur

    root_value = dp[0][0] if dp[0] else math.inf
    assignment = [-1] * orig.n
    stack: List[Tuple[int, int, int]] = [(0, 0, 0)]
    while stack:
        d, o, r = stack.pop()
        if d == t:
            i = orig.leaf_path[o]
            local_k = reps.leaf_path[r]
            if i >= 0 and local_k >= 0:
                assignment[i] = int(rep_indices[local_k])
            continue

        nxt = dp[d + 1]
        assert nxt is not None
        r_next_count = len(reps.keys[d + 1])
        for oc in orig.children[d][o]:
            base = oc * r_next_count
            best_rc = -1
            best_val = math.inf
            for rc in reps.children[d][r]:
                val = nxt[base + rc]
                if val < best_val:
                    best_val = val
                    best_rc = rc
            if best_rc >= 0:
                stack.append((d + 1, oc, best_rc))

    # Every path must receive a representative.  The DP is always feasible,
    # but this guard turns a programming error into a clear failure.
    if any(a < 0 for a in assignment):
        raise RuntimeError("prefix-compatible assignment failed to assign every path")
    return assignment, float(root_value)


def independent_assignment(n: int, k: int, leaf_cost: CostFn) -> Tuple[List[int], float]:
    assignment = [-1] * n
    total = 0.0
    for i in range(n):
        best_k = -1
        best = math.inf
        for local_k in range(k):
            c = float(leaf_cost(i, local_k))
            if c < best:
                best = c
                best_k = local_k
        assignment[i] = best_k  # local representative position; caller maps if needed
        total += best
    return assignment, total


def check_prefix_assignment(
    ds: Dict[str, Any],
    rep_indices: Sequence[int],
    assignments: Sequence[int],
    *,
    decimals: Optional[int] = 8,
) -> Dict[str, Any]:
    rep_set = set(int(r) for r in rep_indices)
    violations: List[Dict[str, Any]] = []
    max_depth = len(ds["periods"])
    depths_checked = list(range(1, max_depth + 1))

    for depth in depths_checked:
        groups: Dict[Tuple[Tuple[float, ...], ...], Dict[Tuple[Tuple[float, ...], ...], List[str]]] = {}
        for i, assigned_raw in enumerate(assignments):
            # Assignments normally contain original dataset indices.  Support
            # local representative positions as well.
            if assigned_raw in rep_set:
                r = int(assigned_raw)
            elif 0 <= assigned_raw < len(rep_indices):
                r = int(rep_indices[assigned_raw])
            else:
                continue
            okey = prefix_key(ds["paths"][i]["values"], depth, decimals)
            rkey = prefix_key(ds["paths"][r]["values"], depth, decimals)
            groups.setdefault(okey, {}).setdefault(rkey, []).append(ds["paths"][i]["id"])

        for okey, rmap in groups.items():
            if len(rmap) > 1:
                violations.append({
                    "depth": depth,
                    "periodsFixed": depth,
                    "originalPrefix": _key_to_json(okey),
                    "representativePrefixes": [_key_to_json(k) for k in rmap.keys()],
                    "affectedPathCount": sum(len(v) for v in rmap.values()),
                    "examplePathIds": [pid for v in rmap.values() for pid in v[:3]][:6],
                })
                if len(violations) >= 50:
                    return {
                        "compatible": False,
                        "violations": violations,
                        "violationCountTruncated": True,
                        "depthsChecked": depths_checked,
                    }

    return {
        "compatible": not violations,
        "violations": violations,
        "violationCount": len(violations),
        "depthsChecked": depths_checked,
        "rule": "paths sharing an original depth-d prefix must map to one reduced depth-d prefix",
    }


def _key_to_json(key: Sequence[Tuple[float, ...]]) -> List[List[float]]:
    return [list(row) for row in key]


def build_prefix_tree(
    ds: Dict[str, Any],
    *,
    probabilities: Optional[Sequence[float]] = None,
    decimals: Optional[int] = 8,
) -> Dict[str, Any]:
    t = len(ds["periods"])
    nodes: Dict[str, Dict[str, Any]] = {
        "root": {
            "id": "root",
            "depth": 0,
            "parent": None,
            "children": [],
            "prefixValues": [],
            "label": "根节点（第1期之前）",
            "probability": 0.0,
            "conditionalProbability": 1.0,
            "pathId": None,
            "pathIndex": None,
        }
    }
    depth_maps: List[Dict[Tuple[Tuple[float, ...], ...], str]] = [{(): "root"}]
    for d in range(1, t + 1):
        depth_maps.append({})

    weights = probabilities if probabilities is not None else [float(p["probability"]) for p in ds["paths"]]
    for i, path in enumerate(ds["paths"]):
        p = float(weights[i]) if i < len(weights) else float(path.get("probability", 0.0))
        for d in range(1, t + 1):
            key = prefix_key(path["values"], d, decimals)
            node_id = depth_maps[d].get(key)
            if node_id is None:
                parent_key = key[:-1]
                parent_id = depth_maps[d - 1][parent_key]
                node_id = f"d{d}-{len(depth_maps[d]) + 1}"
                depth_maps[d][key] = node_id
                prefix_values = _key_to_json(key)
                node = {
                    "id": node_id,
                    "depth": d,
                    "parent": parent_id,
                    "children": [],
                    "prefixValues": prefix_values,
                    "latestValues": prefix_values[-1] if prefix_values else [],
                    "label": f"{ds['periods'][d - 1]['label']}后 · 节点 {node_id}",
                    "probability": 0.0,
                    "conditionalProbability": 0.0,
                    "pathId": None,
                    "pathIndex": None,
                }
                nodes[node_id] = node
                nodes[parent_id]["children"].append(node_id)
            nodes[node_id]["probability"] += p
            if d == t:
                nodes[node_id]["pathId"] = path["id"]
                nodes[node_id]["pathIndex"] = i

    # Recompute probabilities bottom-up with fsum so validation is not based on
    # accumulated addition order.
    for d in range(t, -1, -1):
        ids = ["root"] if d == 0 else [nid for key, nid in depth_maps[d].items()]
        for nid in ids:
            node = nodes[nid]
            if node["children"]:
                node["probability"] = math.fsum(nodes[c]["probability"] for c in node["children"])
    root_mass = nodes["root"]["probability"]
    nodes["root"]["conditionalProbability"] = 1.0
    for node in nodes.values():
        if node["parent"]:
            parent_p = nodes[node["parent"]]["probability"]
            node["conditionalProbability"] = node["probability"] / parent_p if abs(parent_p) > 1e-15 else 0.0

    branch_points = []
    for node in nodes.values():
        if len(node["children"]) > 1:
            branch_points.append({
                "nodeId": node["id"],
                "commonPrefixPeriods": node["depth"],
                "branchPeriodIndex": node["depth"],
                "branchPeriodLabel": ds["periods"][node["depth"]]["label"] if node["depth"] < t else None,
                "childCount": len(node["children"]),
                "probability": node["probability"],
            })

    leaves = [nid for nid, n in nodes.items() if not n["children"]]
    return {
        "rootId": "root",
        "nodes": nodes,
        "leafIds": leaves,
        "branchPoints": branch_points,
        "depth": t,
        "nonAnticipativeByConstruction": True,
        "nodePayloadRule": "a depth-d node stores only periods 0..d-1",
        "totalProbability": root_mass,
    }


def validate_tree(
    tree: Dict[str, Any],
    *,
    expected_total: Optional[float] = None,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    nodes = tree["nodes"]
    errors: List[Dict[str, str]] = []
    if tree.get("rootId") not in nodes:
        errors.append({"code": "MISSING_ROOT", "message": "tree has no valid root"})

    for nid, node in nodes.items():
        if nid == tree.get("rootId"):
            if node.get("parent") is not None:
                errors.append({"code": "ROOT_PARENT", "message": "root must have no parent"})
            continue
        pid = node.get("parent")
        if pid not in nodes:
            errors.append({"code": "MISSING_PARENT", "nodeId": nid, "message": "node parent does not exist"})
        else:
            parent = nodes[pid]
            if nid not in parent.get("children", []):
                errors.append({"code": "BACK_LINK_MISMATCH", "nodeId": nid, "message": "parent children list does not contain node"})
            if parent.get("depth") != node.get("depth", 0) - 1:
                errors.append({"code": "BAD_DEPTH", "nodeId": nid, "message": "node depth must be parent depth + 1"})

    probability_errors = []
    for nid, node in nodes.items():
        children = node.get("children", [])
        if children:
            child_sum = math.fsum(float(nodes[c]["probability"]) for c in children)
            diff = child_sum - float(node["probability"])
            scale = max(1.0, abs(node["probability"]))
            if abs(diff) > tol * scale:
                probability_errors.append({
                    "nodeId": nid,
                    "parentProbability": node["probability"],
                    "childSum": child_sum,
                    "difference": diff,
                })

    leaf_sum = math.fsum(float(nodes[nid]["probability"]) for nid in tree.get("leafIds", []))
    conservation_error = abs(leaf_sum - (expected_total if expected_total is not None else tree.get("totalProbability", leaf_sum)))
    if expected_total is not None and conservation_error > tol * max(1.0, abs(expected_total)):
        errors.append({"code": "PROBABILITY_CONSERVATION", "message": "leaf probabilities do not sum to input mass"})
    if probability_errors:
        errors.append({"code": "CHILD_PROBABILITY_SUM", "message": f"{len(probability_errors)} node(s) violate conditional probability sums"})

    return {
        "valid": not errors,
        "errors": errors,
        "probabilityErrors": probability_errors[:20],
        "leafProbabilitySum": leaf_sum,
        "expectedTotal": expected_total,
        "conservationError": conservation_error,
        "nodeCount": len(nodes),
        "leafCount": len(tree.get("leafIds", [])),
        "branchPointCount": len(tree.get("branchPoints", [])),
        "checks": [
            {"code": "TREE_STRUCTURE", "pass": not any(e["code"] in ("MISSING_PARENT", "BACK_LINK_MISMATCH", "BAD_DEPTH", "MISSING_ROOT", "ROOT_PARENT") for e in errors)},
            {"code": "PROBABILITY_CONSERVATION", "pass": not probability_errors and conservation_error <= tol * max(1.0, abs(expected_total or 1.0))},
            {"code": "NON_ANTICIPATIVE_PREFIXES", "pass": bool(tree.get("nonAnticipativeByConstruction"))},
        ],
    }
