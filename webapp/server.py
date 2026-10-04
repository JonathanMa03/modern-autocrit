"""Dependency-light HTTP server for structured eligibility criteria."""
from __future__ import annotations
import json, mimetypes, threading, webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from backend.api.app_state import create_app_state
from backend.services.eav_workflow import EavWorkflowService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"

class BrowserApplication:
    def __init__(self) -> None:
        self.workflow = EavWorkflowService(create_app_state(), PROJECT_ROOT / "runtime" / "criteria", PROJECT_ROOT / "config" / "base_criterion_library.json")
    def dispatch(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        if method == "GET" and path == "/api/health": return 200, {"status": "ok"}
        if method == "POST" and path == "/api/extractions":
            return 202, self.workflow.submit(input_mode=str(body.get("input_mode") or ""), trial_id=str(body.get("trial_id") or ""), protocol_text=str(body.get("protocol_text") or ""), protocol_pdf=str(body.get("protocol_pdf") or ""))
        if method == "GET" and path.startswith("/api/extractions/"): return 200, self.workflow.get(path.rsplit("/", 1)[-1])
        if method == "POST" and path.startswith("/api/reviews/"):
            rows = body.get("rows")
            if not isinstance(rows, list): raise ValueError("Review rows must be an array.")
            return 200, self.workflow.apply_review(path.rsplit("/", 1)[-1], rows)
        if method == "GET" and path == "/api/library": return 200, self.workflow.library()
        return 404, {"error": "Not found"}

def _handler(application: BrowserApplication):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None: self._handle("GET")
        def do_POST(self) -> None: self._handle("POST")
        def _handle(self, method: str) -> None:
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/"):
                    status, payload = application.dispatch(method, path, self._json_body() if method == "POST" else {}); self._send_json(status, payload)
                elif method == "GET": self._send_static(path)
                else: self._send_json(405, {"error": "Method not allowed"})
            except KeyError as exc: self._send_json(404, {"error": f"Unknown resource: {exc}"})
            except ValueError as exc: self._send_json(400, {"error": str(exc)})
            except Exception as exc: self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
        def _json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 75_000_000: raise ValueError("Request body is too large (75 MB maximum).")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict): raise ValueError("Request body must be an object.")
            return payload
        def _send_json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8"); self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
        def _send_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/"); target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT not in target.parents or not target.is_file(): self.send_error(HTTPStatus.NOT_FOUND); return
            data = target.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
        def log_message(self, fmt: str, *args: Any) -> None: return
    return Handler

def run_browser_app(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    app = BrowserApplication(); server = ThreadingHTTPServer((host, port), _handler(app)); server.daemon_threads = True; url = f"http://{host}:{port}"
    print(f"Automated Eligibility Criteria Extraction running at {url}")
    if open_browser: threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try: server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt: pass
    finally: server.shutdown(); server.server_close()
