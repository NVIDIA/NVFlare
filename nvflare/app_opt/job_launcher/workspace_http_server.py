# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTPS server that delivers workspace files to Kubernetes job pods.

One :class:`WorkspaceHTTPServer` per parent process, shared across jobs.
Each job gets an unguessable capability URL (256-bit random token).

    GET  /<token>  → download workspace ZIP
    POST /<token>  → upload results ZIP

Security: TLS 1.2+ for confidentiality/integrity, capability URL for access control.
``startup/`` and ``local/`` are delivered via k8s Secret and ConfigMap, never over HTTP.
"""

import io
import logging
import os
import secrets
import ssl
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

ENV_WORKSPACE_URL = "NVFL_WORKSPACE_URL"


class _Handler(BaseHTTPRequestHandler):
    """Minimal handler — accesses shared state via self.server.ws."""

    def do_GET(self):
        data = self.server.ws.jobs.get(self.path.strip("/"))
        if data is None:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        token = self.path.strip("/")
        ws = self.server.ws
        if token not in ws.jobs:
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(ws.workspace_root, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            zf.extractall(ws.workspace_root)
        ws.jobs.pop(token, None)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_):
        pass


class WorkspaceHTTPServer:
    """Shared HTTPS server for workspace delivery and results collection."""

    def __init__(self, cert_file: str = "", key_file: str = ""):
        self._cert_file = cert_file
        self._key_file = key_file
        self._port: int = 0
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._scheme = "http"
        self._stop = threading.Event()
        self.jobs: dict[str, bytes] = {}  # url_token -> zip_bytes
        self.workspace_root = ""

    @property
    def port(self) -> int:
        return self._port

    def start(self, workspace_root: str, bind: str = "0.0.0.0") -> int:
        self.workspace_root = workspace_root
        srv = HTTPServer((bind, 0), _Handler)
        srv.ws = self  # back-reference for handler
        self._port = srv.server_address[1]

        if self._cert_file and self._key_file:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(self._cert_file, self._key_file)
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
            self._scheme = "https"

        self._server = srv
        self._thread = threading.Thread(target=self._serve, daemon=True, name="ws-http")
        self._thread.start()
        return self._port

    def add_job(self, job_id: str, workspace_root: str) -> str:
        """Register a job. Returns an unguessable URL token."""
        token = secrets.token_urlsafe(32)
        self.jobs[token] = _zip_workspace(workspace_root, job_id)
        return token

    def remove_job(self, token: str) -> None:
        self.jobs.pop(token, None)

    def get_url(self, host: str, token: str) -> str:
        return f"{self._scheme}://{host}:{self._port}/{token}"

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _serve(self) -> None:
        self._server.timeout = 1.0
        while not self._stop.is_set():
            self._server.handle_request()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zip_workspace(workspace_root: str, job_id: str) -> bytes:
    """ZIP the job directory from *workspace_root* (excludes startup/ and local/)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        src = os.path.join(workspace_root, job_id)
        if os.path.isdir(src):
            for dirpath, _dirs, files in os.walk(src):
                for fname in files:
                    abs_path = os.path.join(dirpath, fname)
                    zf.write(abs_path, os.path.relpath(abs_path, workspace_root))
    return buf.getvalue()


def _build_ssl_context(ca_cert: str) -> ssl.SSLContext:
    # URL is a 256-bit capability token; parent is addressed by pod IP which is
    # not in the NVFlare cert SAN. Validate CA chain for encryption/integrity
    # but skip hostname match.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    if ca_cert:
        ctx.load_verify_locations(ca_cert)
    else:
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Pod-side functions (read NVFL_WORKSPACE_URL from env; no-op when unset)
# ---------------------------------------------------------------------------


def download_workspace(dest: str) -> None:
    """GET the workspace ZIP and extract into *dest*. No-op if env var unset."""
    url = os.environ.get(ENV_WORKSPACE_URL)
    if not url:
        return
    import urllib.request

    ca_cert = os.path.join(dest, "startup", "rootCA.pem")
    ctx = _build_ssl_context(ca_cert if os.path.exists(ca_cert) else "")
    os.makedirs(dest, exist_ok=True)
    logger.info("Downloading workspace from %s", url)
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest)


def upload_results(workspace_root: str, job_id: str) -> None:
    """ZIP the job directory and POST it. No-op if env var unset."""
    url = os.environ.get(ENV_WORKSPACE_URL)
    if not url:
        return
    import urllib.request

    run_dir = os.path.join(workspace_root, job_id)
    if not os.path.isdir(run_dir):
        return

    ca_cert = os.path.join(workspace_root, "startup", "rootCA.pem")
    ctx = _build_ssl_context(ca_cert if os.path.exists(ca_cert) else "")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirs, files in os.walk(run_dir):
            for fname in files:
                abs_path = os.path.join(dirpath, fname)
                zf.write(abs_path, os.path.relpath(abs_path, workspace_root))
    data = buf.getvalue()

    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/zip"})
    urllib.request.urlopen(req, timeout=120, context=ctx).close()
