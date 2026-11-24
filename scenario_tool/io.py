"""Import helpers for JSON and simple wide/long CSV scenario formats."""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List, Tuple


def parse_import_text(filename: str, text: str) -> Dict[str, Any]:
    name = (filename or "").lower()
    stripped = text.lstrip()
    if name.endswith(".json") or stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(text)
        if isinstance(payload, list):
            payload = {"paths": payload}
        if not isinstance(payload, dict):
            raise ValueError("JSON import must be an object or path array")
        return payload
    return parse_csv(text)


def _split_period_dimension(header: str) -> Tuple[int, str]:
    h = header.strip()
    # t0_demand, T-0 demand, 0_demand, period0_demand
    m = re.match(r"^(?:period|p|t)?[_\s:-]*0*(\d+)[_\s:.-]+(.+)$", h, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2).strip()
    # demand_t0, demand_0, demand period 0
    m = re.match(r"^(.+?)[_\s:.-]+(?:period|p|t)?[_\s:-]*0*(\d+)$", h, flags=re.IGNORECASE)
    if m:
        return int(m.group(2)), m.group(1).strip()
    raise ValueError(f"cannot parse period/dimension column: {header}")


def parse_csv(text: str) -> Dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV is empty")
    headers = [h.strip() for h in reader.fieldnames if h]
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")

    lower = {h.lower(): h for h in headers}
    is_long = any(h.lower() in ("period", "t") for h in headers) and any(
        h.lower() in ("dimension", "dim", "variable", "metric") for h in headers
    ) and any(h.lower() in ("value", "val") for h in headers)

    if is_long:
        return _parse_long_csv(rows, lower)

    id_col = lower.get("id") or lower.get("scenario") or lower.get("scenarioid")
    prob_col = lower.get("probability") or lower.get("p") or lower.get("weight")
    tag_col = lower.get("tags") or lower.get("tag")

    columns = []
    for h in headers:
        if h in (id_col, prob_col, tag_col) or h.lower() in ("id", "probability", "p", "weight", "tags", "tag", "name"):
            continue
        period, dim = _split_period_dimension(h)
        columns.append((h, period, dim))
    if not columns:
        raise ValueError("CSV needs columns such as t0_demand,t0_freight,...")

    periods = sorted({p for _, p, _ in columns})
    dims: List[str] = []
    for _, _, d in columns:
        if d not in dims:
            dims.append(d)

    paths = []
    for i, row in enumerate(rows):
        values = [[0.0 for _ in dims] for _ in periods]
        for header, p, d in columns:
            try:
                values[periods.index(p)][dims.index(d)] = float(row[header])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric value in column {header}, row {i + 2}") from exc
        prob = float(row[prob_col]) if prob_col and row.get(prob_col) not in (None, "") else 1.0 / len(rows)
        tags = []
        if tag_col and row.get(tag_col):
            tags = [x.strip() for x in re.split(r"[|;,]", row[tag_col]) if x.strip()]
        paths.append({
            "id": (row.get(id_col) if id_col else None) or f"CSV{i + 1:05d}",
            "probability": prob,
            "values": values,
            "tags": tags,
        })

    return {
        "name": "CSV imported scenario set",
        "periods": [f"T{p + 1}" for p in periods],
        "dimensions": [{"key": d, "label": d} for d in dims],
        "paths": paths,
    }


def _parse_long_csv(rows: List[Dict[str, str]], lower: Dict[str, str]) -> Dict[str, Any]:
    id_col = lower.get("id") or lower.get("scenario") or "id"
    prob_col = lower.get("probability") or lower.get("p") or lower.get("weight")
    period_col = lower.get("period") or lower.get("t")
    dim_col = lower.get("dimension") or lower.get("dim") or lower.get("variable")
    value_col = lower.get("value") or lower.get("val")
    tags_col = lower.get("tags") or lower.get("tag")

    grouped: Dict[str, Dict[str, Any]] = {}
    dims: List[str] = []
    period_set = set()
    for row in rows:
        sid = row.get(id_col) or f"ROW{len(grouped) + 1}"
        item = grouped.setdefault(sid, {"probability": None, "cells": {}, "tags": []})
        if prob_col and row.get(prob_col):
            item["probability"] = float(row[prob_col])
        if tags_col and row.get(tags_col):
            item["tags"] = [x.strip() for x in re.split(r"[|;,]", row[tags_col]) if x.strip()]
        period = int(float(row[period_col]))
        if period == 0 or (isinstance(period, int) and period >= 1 and str(row[period_col]).strip().startswith("1") and len(str(row[period_col]).strip()) == 1):
            # Accept both 0-based and 1-based period numbers; normalize below.
            pass
        dim = row[dim_col]
        value = float(row[value_col])
        if dim not in dims:
            dims.append(dim)
        period_set.add(period)
        item["cells"][(period, dim)] = value

    periods = sorted(period_set)
    # If labels are 1..T, convert to array positions preserving order.
    period_pos = {p: i for i, p in enumerate(periods)}
    paths = []
    for sid, item in grouped.items():
        values = [[item["cells"].get((p, d), 0.0) for d in dims] for p in periods]
        paths.append({
            "id": sid,
            "probability": item["probability"] if item["probability"] is not None else 1.0 / len(grouped),
            "values": values,
            "tags": item["tags"],
        })
    return {
        "name": "Long CSV imported scenario set",
        "periods": [f"T{p}" for p in periods],
        "dimensions": [{"key": d, "label": d} for d in dims],
        "paths": paths,
    }
