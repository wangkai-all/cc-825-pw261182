"""Scenario import, canonicalization, duplicate merging, and validation.

The module deliberately uses plain JSON-compatible dictionaries so the same
functions can be used from the CLI, HTTP API, and tests without an ORM layer.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

NORMALIZATION_TOL = 1e-6
PROBABILITY_TOL = 1e-9


class DatasetValidationError(ValueError):
    """Raised when a dataset has structural or negative-probability errors."""

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        errors = "; ".join(
            e.get("message", str(e)) for e in report.get("errors", [])
        ) or "invalid scenario dataset"
        super().__init__(errors)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _clean_float(x: Any) -> float:
    y = float(x)
    if y == 0.0:
        return 0.0  # canonicalize -0.0
    return y


def _dimension_specs(payload: Dict[str, Any], first_values: Any) -> List[Dict[str, Any]]:
    raw_dims = payload.get("dimensions")
    dims: List[Dict[str, Any]] = []

    if raw_dims is None:
        if isinstance(first_values, dict):
            raw_dims = list(first_values.keys())
        elif isinstance(first_values, list):
            raw_dims = [f"x{i + 1}" for i in range(len(first_values[0]) if first_values and isinstance(first_values[0], list) else len(first_values))]
        else:
            raw_dims = ["value"]

    for i, item in enumerate(raw_dims):
        if isinstance(item, str):
            dim = {"key": item, "label": item, "unit": "", "scale": None, "weight": 1.0}
        elif isinstance(item, dict):
            key = str(item.get("key") or item.get("id") or item.get("name") or f"x{i + 1}")
            scale = item.get("scale")
            dim = {
                "key": key,
                "label": str(item.get("label") or item.get("name") or key),
                "unit": str(item.get("unit") or ""),
                "scale": None if scale is None else float(scale),
                "weight": float(item.get("weight", 1.0)),
            }
        else:
            raise DatasetValidationError(
                {"errors": [{"code": "BAD_DIMENSION", "message": f"dimension {i} is invalid"}]}
            )
        if not math.isfinite(dim["weight"]) or dim["weight"] < 0:
            dim["weight"] = 1.0
        dims.append(dim)

    if not dims:
        dims = [{"key": "value", "label": "Value", "unit": "", "scale": None, "weight": 1.0}]
    return dims


def _canonical_values(values: Any, dims: List[Dict[str, Any]], path_index: int) -> List[List[float]]:
    keys = [d["key"] for d in dims]
    if not isinstance(values, list) or not values:
        raise DatasetValidationError(
            {"errors": [{"code": "BAD_VALUES", "pathIndex": path_index, "message": "values must be a non-empty list of periods"}]}
        )

    out: List[List[float]] = []
    expected_d = len(keys)
    for t, row in enumerate(values):
        if isinstance(row, dict):
            missing = [k for k in keys if k not in row]
            if missing:
                raise DatasetValidationError(
                    {"errors": [{
                        "code": "MISSING_DIMENSION",
                        "pathIndex": path_index,
                        "period": t,
                        "message": f"period {t} is missing {', '.join(missing)}",
                    }]}
                )
            raw = [row[k] for k in keys]
        elif isinstance(row, (list, tuple)):
            raw = list(row)
        elif expected_d == 1 and _is_number(row):
            raw = [row]
        else:
            raise DatasetValidationError(
                {"errors": [{
                    "code": "BAD_PERIOD",
                    "pathIndex": path_index,
                    "period": t,
                    "message": "period values must be an object, array, or number",
                }]}
            )

        if len(raw) != expected_d:
            raise DatasetValidationError(
                {"errors": [{
                    "code": "BAD_DIMENSION_COUNT",
                    "pathIndex": path_index,
                    "period": t,
                    "message": f"expected {expected_d} dimensions, got {len(raw)}",
                }]}
            )
        clean: List[float] = []
        for d_idx, x in enumerate(raw):
            if not _is_number(x):
                raise DatasetValidationError(
                    {"errors": [{
                        "code": "NON_NUMERIC_VALUE",
                        "pathIndex": path_index,
                        "period": t,
                        "dimension": keys[d_idx],
                        "message": "all path values must be finite numbers",
                    }]}
                )
            clean.append(_clean_float(x))
        out.append(clean)
    return out


def canonicalize_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate schema and convert all path values to period-by-dimension arrays."""
    if not isinstance(payload, dict):
        raise DatasetValidationError({"errors": [{"code": "BAD_PAYLOAD", "message": "request body must be a JSON object"}]})

    raw_paths = payload.get("paths") or payload.get("scenarios")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise DatasetValidationError({"errors": [{"code": "NO_PATHS", "message": "dataset must contain a non-empty paths array"}]})

    first = raw_paths[0]
    if not isinstance(first, dict):
        raise DatasetValidationError({"errors": [{"code": "BAD_PATH", "pathIndex": 0, "message": "each path must be an object"}]})
    first_values = first.get("values") if "values" in first else first.get("data")
    dims = _dimension_specs(payload, first_values)

    periods_meta = payload.get("periods")
    paths: List[Dict[str, Any]] = []
    t_count: Optional[int] = None
    errors: List[Dict[str, Any]] = []

    for i, raw in enumerate(raw_paths):
        if not isinstance(raw, dict):
            errors.append({"code": "BAD_PATH", "pathIndex": i, "message": "path must be an object"})
            continue
        path_id = str(raw.get("id") or raw.get("name") or f"P{i + 1:05d}")
        try:
            p = raw.get("probability", raw.get("p", raw.get("weight")))
            if not _is_number(p):
                errors.append({"code": "BAD_PROBABILITY", "pathId": path_id, "message": "probability must be a finite number"})
                p = 0.0
            else:
                p = float(p)
            values = _canonical_values(raw.get("values", raw.get("data")), dims, i)
        except DatasetValidationError as exc:
            errors.extend(exc.report.get("errors", []))
            continue

        if t_count is None:
            t_count = len(values)
        elif len(values) != t_count:
            errors.append({
                "code": "BAD_PERIOD_COUNT",
                "pathId": path_id,
                "message": f"expected {t_count} periods, got {len(values)}",
            })
            continue

        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        paths.append({
            "id": path_id,
            "probability": _clean_float(p),
            "values": values,
            "tags": [str(x) for x in tags],
            "sourceIds": list(raw.get("sourceIds") or [path_id]),
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        })

    if errors:
        raise DatasetValidationError({"errors": errors})

    assert t_count is not None
    if periods_meta and isinstance(periods_meta, list) and len(periods_meta) == t_count:
        periods = []
        for t, item in enumerate(periods_meta):
            if isinstance(item, dict):
                periods.append({"index": t, "label": str(item.get("label") or item.get("name") or f"T{t + 1}")})
            else:
                periods.append({"index": t, "label": str(item)})
    else:
        periods = [{"index": t, "label": f"T{t + 1}"} for t in range(t_count)]

    return {
        "name": str(payload.get("name") or "Unnamed scenario set"),
        "description": str(payload.get("description") or ""),
        "periods": periods,
        "dimensions": dims,
        "paths": paths,
        "planning": payload.get("planning") if isinstance(payload.get("planning"), dict) else {},
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "kind": "scenario-paths",
    }


def path_identity(path: Dict[str, Any]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(row) for row in path["values"])


def merge_duplicate_paths(ds: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Merge exactly identical paths and sum their probabilities.

    This is intentionally performed before zero-probability and normalization
    checks so that a duplicated mass is never counted as two scenarios.
    """
    groups: Dict[Tuple[Tuple[float, ...], ...], Dict[str, Any]] = {}
    order: List[Tuple[Tuple[float, ...], ...]] = []
    merge_report: List[Dict[str, Any]] = []

    for path in ds["paths"]:
        key = path_identity(path)
        if key not in groups:
            item = {
                "id": path["id"],
                "probability": path["probability"],
                "values": path["values"],
                "tags": list(path.get("tags", [])),
                "sourceIds": list(path.get("sourceIds", [path["id"]])),
                "metadata": dict(path.get("metadata", {})),
                "duplicateCount": 1,
            }
            groups[key] = item
            order.append(key)
        else:
            item = groups[key]
            before = item["probability"]
            item["probability"] = _clean_float(before + path["probability"])
            item["duplicateCount"] += 1
            item["sourceIds"].extend(path.get("sourceIds", [path["id"]]))
            for tag in path.get("tags", []):
                if tag not in item["tags"]:
                    item["tags"].append(tag)
            merge_report.append({
                "keptId": item["id"],
                "mergedId": path["id"],
                "probabilityBefore": path["probability"],
                "probabilityAfterMerge": item["probability"],
            })

    merged_paths = [groups[key] for key in order]
    out = dict(ds)
    out["paths"] = merged_paths
    return out, merge_report


def check_probabilities(ds: Dict[str, Any], tol: float = NORMALIZATION_TOL) -> Dict[str, Any]:
    probs = [float(p["probability"]) for p in ds["paths"]]
    total = math.fsum(probs)
    zero_ids = [p["id"] for p in ds["paths"] if p["probability"] == 0.0]
    negative = [
        {"pathId": p["id"], "probability": p["probability"]}
        for p in ds["paths"] if p["probability"] < 0.0
    ]
    nearly_zero = [
        {"pathId": p["id"], "probability": p["probability"]}
        for p in ds["paths"] if 0.0 < abs(p["probability"]) < PROBABILITY_TOL
    ]
    normalized = abs(total - 1.0) <= tol
    return {
        "pathCount": len(ds["paths"]),
        "sum": total,
        "normalized": normalized,
        "normalizationTolerance": tol,
        "differenceFromOne": total - 1.0,
        "zeroProbabilityCount": len(zero_ids),
        "zeroProbabilityIds": zero_ids[:100],
        "zeroProbabilityTruncated": len(zero_ids) > 100,
        "nearlyZero": nearly_zero[:100],
        "negativeProbabilities": negative,
        "valid": not negative and math.isfinite(total),
        "checks": [
            {
                "code": "ZERO_PROBABILITY",
                "status": "warning" if zero_ids else "pass",
                "count": len(zero_ids),
                "message": (
                    f"{len(zero_ids)} path(s) have exactly zero probability; they are inspected but excluded from representative candidacy."
                    if zero_ids else "No exactly-zero probabilities."
                ),
            },
            {
                "code": "UNNORMALIZED_PROBABILITY",
                "status": "pass" if normalized else "warning",
                "sum": total,
                "message": (
                    f"Probabilities sum to {total:.12g}, within tolerance of 1."
                    if normalized else f"Probabilities sum to {total:.12g}, not 1 (Δ={total - 1:+.6g})."
                ),
            },
            {
                "code": "NEGATIVE_PROBABILITY",
                "status": "error" if negative else "pass",
                "count": len(negative),
                "message": "Probabilities must be non-negative." if negative else "All probabilities are non-negative.",
            },
        ],
    }


def normalize_probabilities(ds: Dict[str, Any]) -> Dict[str, Any]:
    total = math.fsum(float(p["probability"]) for p in ds["paths"])
    if not math.isfinite(total) or total <= 0:
        raise DatasetValidationError({
            "errors": [{"code": "CANNOT_NORMALIZE", "message": "probability mass must be positive before normalization"}]
        })
    out = dict(ds)
    out["paths"] = [dict(p, probability=_clean_float(p["probability"] / total)) for p in ds["paths"]]
    out["probabilityNormalizedFromSum"] = total
    return out


def prepare_dataset(
    payload: Dict[str, Any],
    *,
    normalize: bool = False,
    drop_zero: bool = False,
    strict: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Canonicalize, merge duplicates, and independently validate probabilities."""
    ds = canonicalize_dataset(payload)
    original_count = len(ds["paths"])
    ds, duplicates = merge_duplicate_paths(ds)
    duplicate_reduction = original_count - len(ds["paths"])

    prob_report = check_probabilities(ds)
    errors = []
    if prob_report["negativeProbabilities"]:
        errors.append({"code": "NEGATIVE_PROBABILITY", "message": "negative probabilities are not allowed"})

    actions: List[str] = []
    if normalize and not prob_report["normalized"]:
        ds = normalize_probabilities(ds)
        actions.append("normalized")
        prob_report = check_probabilities(ds)
    if drop_zero and prob_report["zeroProbabilityCount"]:
        ds = dict(ds)
        ds["paths"] = [p for p in ds["paths"] if p["probability"] != 0.0]
        actions.append("dropped-zero")
        prob_report = check_probabilities(ds)

    report = {
        "originalPathCount": original_count,
        "pathCountAfterDuplicateMerge": len(ds["paths"]),
        "duplicateRowsMerged": duplicate_reduction,
        "duplicateGroups": duplicates[:200],
        "duplicateGroupCount": len(duplicates),
        "probabilities": prob_report,
        "actions": actions,
        "errors": errors,
        "valid": not errors and prob_report["valid"],
    }
    if strict and errors:
        raise DatasetValidationError(report)
    return ds, report


def active_paths(ds: Dict[str, Any]) -> Tuple[List[int], float]:
    """Indices eligible to be representatives and the positive mass they carry."""
    idx = [i for i, p in enumerate(ds["paths"]) if p["probability"] > 0.0]
    mass = math.fsum(ds["paths"][i]["probability"] for i in idx)
    return idx, mass
