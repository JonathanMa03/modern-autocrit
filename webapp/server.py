"""Dependency-light local HTTP server for Modern AutoCrit."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.api.app_state import create_app_state
from backend.services.eligcrit_extraction import EligCritExtractionService
from backend.statistics.clinical_distance import clinical_distance_penalty
from backend.criteria_processor.eva_library import EvaLibraryRepository
from backend.criteria_processor.segmentation import iter_eligibility_source_items
from backend.criteria_processor.value_normalization import normalize_structured_value
from backend.ctg_parser.parser import parse_ctgov_study


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"


class BrowserApplication:
    def __init__(self) -> None:
        self.state = create_app_state()
        self.extraction = EligCritExtractionService(self.state)
        self.library = EvaLibraryRepository(
            PROJECT_ROOT / "config" / "eligibility_eva_library.json",
            snapshot_directory=PROJECT_ROOT / "runtime" / "eligcrit" / "library_snapshots",
        )

    def resume(self) -> int:
        return self.extraction.resume_incomplete()

    def dispatch(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        if method == "GET" and path == "/api/health":
            return 200, {"status": "ok", "interface": "browser"}
        if method == "GET" and path == "/api/library":
            return 200, {
                "revision": self.library.revision,
                "sha256": self.library.sha256,
                "entities": [item.model_dump(mode="json") for item in self.library.entities.values()],
                "attributes": [item.model_dump(mode="json") for item in self.library.attributes.values()],
            }
        if method == "POST" and path == "/api/segment":
            rows = iter_eligibility_source_items(str(body.get("eligibility_text") or ""))
            return 200, {"items": [item.model_dump(mode="json") for item in rows]}
        if method == "POST" and path == "/api/normalize-value":
            allowed = {
                "raw_value", "attribute_id", "canonical_attribute", "value_schema",
                "canonical_unit", "canonical_value", "context_text", "criterion_type",
            }
            values = {key: body.get(key) for key in allowed}
            return 200, normalize_structured_value(**values)
        if method == "POST" and path == "/api/distance":
            result = clinical_distance_penalty(
                float(body["value"]),
                target=body["target"],
                numerical_penalty={
                    "parameters": {
                        "full_relevance_tolerance": float(body.get("tolerance", 0)),
                        "clinical_distance_scale": float(body.get("scale", 10)),
                    }
                },
                condition_weight=float(body.get("weight", 1)),
            )
            return 200, result
        if method == "POST" and path == "/api/ctgov/parse":
            return 200, parse_ctgov_study(body).model_dump(mode="json")
        if method == "POST" and path == "/api/jobs/extract":
            return 202, self.extraction.submit(
                trial_id=str(body.get("trial_id") or ""),
                eligibility_text=str(body.get("eligibility_text") or ""),
            )
        if method == "GET" and path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 3 else ""
            job = self.extraction.get(job_id)
            if len(parts) == 4 and parts[3] == "result":
                audit_path = job.get("audit_path")
                if not audit_path:
                    return 409, {"error": "Job has no completed audit result."}
                return 200, json.loads(Path(audit_path).read_text(encoding="utf-8"))
            if len(parts) == 4 and parts[3] == "cleaned":
                audit_path = job.get("audit_path")
                if not audit_path:
                    return 409, {"error": "Job has no completed audit result."}
                audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
                return 200, {
                    "trial_id": audit.get("trial_id"),
                    "review_status": audit.get("review_status"),
                    "criteria": [_cleaned_item(item) for item in audit.get("items", [])],
                }
            return 200, job
        return 404, {"error": "Not found"}


def _handler(application: BrowserApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutomatedEligibilityExtraction/1.0"

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def _handle(self, method: str) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/"):
                try:
                    body = self._json_body() if method == "POST" else {}
                    status, payload = application.dispatch(method, path, body)
                except KeyError as exc:
                    status, payload = 404, {"error": f"Not found: {exc.args[0]}"}
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    status, payload = 400, {"error": str(exc)}
                except Exception as exc:
                    status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
                self._send_json(status, payload)
                return
            self._send_static(path)

        def _json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 5_000_000:
                raise ValueError("Request body is too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def _cleaned_item(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("categorical_value")
    if item.get("value_type") == "Numerical":
        value = item.get("numerical_value")
    return {
        "criteria": item.get("criteria"),
        "source": item.get("item"),
        "entity": item.get("entity_name"),
        "attribute": item.get("attribute_name"),
        "attribute_id": item.get("attribute_id"),
        "value": value,
        "unit": item.get("unit"),
        "confidence": item.get("confidence"),
        "mapping": item.get("mapping_decision"),
        "mapping_review": (item.get("mapping_review") or {}).get("final_decision"),
        "review_status": item.get("review_status"),
    }


def run_browser_app(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    application = BrowserApplication()
    application.resume()
    server = ThreadingHTTPServer((host, port), _handler(application))
    url = f"http://{host}:{port}"
    print(f"Modern AutoCrit is running at {url}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
