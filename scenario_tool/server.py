"""Zero-dependency HTTP server and JSON API for the scenario reduction workbench."""
from __future__ import annotations

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .dataset import DatasetValidationError, prepare_dataset
from .io import parse_import_text
from .pipeline import run_analysis, run_exact_check, run_pareto, run_recovery_check
from .sample import generate_fictional_dataset, toy_dataset

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


class ScenarioHandler(BaseHTTPRequestHandler):
    server_version = "ScenarioTool/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep test/server output concise; errors are still returned to clients.
        if os.environ.get("SCENARIO_TOOL_HTTP_DEBUG"):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, route: str) -> None:
        if route not in STATIC:
            self._send_json(404, {"error": "not found"})
            return
        filename, content_type = STATIC[route]
        path = WEB_DIR / filename
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": f"missing web asset {filename}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route.startswith("/api/"):
            try:
                if route == "/api/health":
                    self._send_json(200, {"ok": True, "service": "scenario-reduction-tool"})
                elif route == "/api/sample":
                    qs = parse_qs(parsed.query)
                    if qs.get("toy", ["0"])[0] in ("1", "true", "yes"):
                        payload = toy_dataset()
                    else:
                        try:
                            size = int(qs.get("size", ["180"])[0])
                        except ValueError:
                            size = 180
                        payload = generate_fictional_dataset(size)
                    self._send_json(200, payload)
                else:
                    self._send_json(404, {"error": "unknown API route"})
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self._send_json(400, {"error": str(exc)})
            return
        self._send_static(route)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            body = self._read_json()
            if route == "/api/import":
                payload = parse_import_text(str(body.get("filename", "import.json")), str(body.get("text", "")))
                ds, report = prepare_dataset(payload)
                self._send_json(200, {"dataset": ds, "validation": report})
            elif route == "/api/validate":
                payload = body.get("dataset") if "dataset" in body else body
                options = body.get("options", {}) if "dataset" in body else {}
                ds, report = prepare_dataset(
                    payload,
                    normalize=bool(options.get("normalize", body.get("normalize", False))),
                    drop_zero=bool(options.get("dropZero", body.get("dropZero", False))),
                )
                self._send_json(200, {"dataset": ds, "validation": report})
            elif route == "/api/analyze":
                dataset = body.get("dataset")
                if dataset is None:
                    raise ValueError("missing dataset")
                options = body.get("options", {})
                result = run_analysis(dataset, options)
                self._send_json(200, result)
            elif route == "/api/pareto":
                dataset = body.get("dataset")
                if dataset is None:
                    raise ValueError("missing dataset")
                options = body.get("options", {})
                weights = body.get("weights")
                result = run_pareto(dataset, options, weights)
                self._send_json(200, result)
            elif route == "/api/exact-check":
                result = run_exact_check(body)
                self._send_json(200, result)
            elif route == "/api/recovery-drill":
                self._send_json(200, run_recovery_check())
            else:
                self._send_json(404, {"error": "unknown API route"})
        except DatasetValidationError as exc:
            self._send_json(422, {"error": str(exc), "validation": exc.report})
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - HTTP boundary
            traceback.print_exc()
            self._send_json(500, {"error": "internal server error", "detail": str(exc)})


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), ScenarioHandler)
    print(f"Scenario reduction workbench running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
