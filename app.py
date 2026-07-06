"""Custom web server for the energy estimator."""
from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.artifacts import get_model_bundle
from src.service import build_app_config, predict_energy


ROOT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT_DIR / "public"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


class EnergyRequestHandler(BaseHTTPRequestHandler):
    server_version = "EnergyForecastingLab/1.0"

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html"}:
            path = PUBLIC_DIR / "index.html"
        else:
            safe_path = self.path.split("?", 1)[0].lstrip("/")
            path = PUBLIC_DIR / safe_path

        try:
            resolved = path.resolve()
            if not str(resolved).startswith(str(PUBLIC_DIR.resolve())):
                raise FileNotFoundError
            size = resolved.stat().st_size
        except (FileNotFoundError, IsADirectoryError):
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/config":
            self._send_json(build_app_config(get_model_bundle()))
            return

        if self.path in {"/", "/index.html"}:
            self._send_file(PUBLIC_DIR / "index.html")
            return

        safe_path = self.path.split("?", 1)[0].lstrip("/")
        self._send_file(PUBLIC_DIR / safe_path)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            result = predict_energy(get_model_bundle(), payload)
            self._send_json(result)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            if not str(resolved).startswith(str(PUBLIC_DIR.resolve())):
                raise FileNotFoundError
            body = resolved.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8501"))
    server = ThreadingHTTPServer((host, port), EnergyRequestHandler)
    print(f"Serving Home Energy Estimator on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
