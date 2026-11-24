"""Command line interface for validation, reduction, checks, and the web server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .io import parse_import_text
from .pipeline import run_analysis, run_exact_check, run_recovery_check
from .server import main as serve_main


def _read_input(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_import_text(Path(path).name, text)


def _write_json(payload: Any, out: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenario-tool", description="Multi-period scenario reduction workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local web workbench")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    validate = sub.add_parser("validate", help="import, merge duplicates, and check probabilities")
    validate.add_argument("input")
    validate.add_argument("--normalize", action="store_true")

    reduce_p = sub.add_parser("reduce", help="reduce paths and compare the small planning model")
    reduce_p.add_argument("input")
    reduce_p.add_argument("--k", type=int, default=12)
    reduce_p.add_argument("--tail-weight", type=float, default=1.0)
    reduce_p.add_argument("--tail-level", type=float, default=0.90)
    reduce_p.add_argument("--tail-guards", type=int, default=2)
    reduce_p.add_argument("--no-prefix-lock", action="store_true")
    reduce_p.add_argument("--normalize", action="store_true")
    reduce_p.add_argument("--out")

    exact = sub.add_parser("exact-check", help="exhaustive small-instance comparison")
    exact.add_argument("--k", type=int, default=3)
    exact.add_argument("--out")

    recovery = sub.add_parser("recovery-drill", help="crash/resume idempotence drill")
    recovery.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        import os
        os.environ["HOST"] = args.host
        os.environ["PORT"] = str(args.port)
        serve_main()
        return 0

    if args.command == "validate":
        from .dataset import prepare_dataset
        payload = _read_input(args.input)
        ds, report = prepare_dataset(payload, normalize=args.normalize)
        _write_json({"dataset": ds, "validation": report}, None)
        return 0 if report["valid"] else 2

    if args.command == "reduce":
        payload = _read_input(args.input)
        result = run_analysis(payload, {
            "k": args.k,
            "tailWeight": args.tail_weight,
            "tailLevel": args.tail_level,
            "tailGuards": args.tail_guards,
            "prefixLocked": not args.no_prefix_lock,
            "normalize": args.normalize,
        })
        if args.out:
            _write_json(result, args.out)
        else:
            r = result["reduction"]
            pc = result["planningComparison"]
            summary = {
                "method": r["method"],
                "representatives": [x["id"] for x in r["representatives"]],
                "probabilityConservation": r["probabilityConservation"],
                "nonAnticipativityCompatible": r["nonAnticipativity"]["compatible"],
                "treeValid": r["treeValidation"]["valid"],
                "averageDistance": r["metrics"]["averageDistance"],
                "cvarError": r["metrics"]["cvarError"],
                "qOriginal": pc["qOriginal"],
                "qReduced": pc["qReduced"],
                "outOfSampleRegret": pc["outOfSampleRegret"],
            }
            _write_json(summary, None)
        return 0

    if args.command == "exact-check":
        result = run_exact_check({"k": args.k})
        _write_json(result, args.out)
        return 0 if result["optimality"]["matchesExhaustiveOptimum"] else 1

    if args.command == "recovery-drill":
        result = run_recovery_check()
        _write_json(result, args.out)
        return 0 if result["final"]["valid"] and result["matchesCleanAllocation"] else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
