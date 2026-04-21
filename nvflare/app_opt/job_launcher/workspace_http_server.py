from __future__ import annotations

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
``startup/`` is delivered via k8s Secret; ``local/`` and the job run dir are
delivered through the per-job workspace bundle.
"""

import io
import logging
import os
import secrets
import shutil
import ssl
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ENV_WORKSPACE_URL = "NVFL_WORKSPACE_URL"
MAX_REQUEST_BODY_SIZE = 1 << 30  # 1 GiB
_STREAM_CHUNK_SIZE = 64 * 1024
_TRANSFER_RETRIES = 3
_TRANSFER_BACKOFF_S = 1.0


class _RequestError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class _JobRecord:
    job_id: str
    workspace_root: str


class _Handler(BaseHTTPRequestHandler):
    """Minimal handler — accesses shared state via self.server.ws."""

    def do_GET(self):
        record = self.server.ws._claim_for_download(self.path.strip("/"))
        if record is None:
            return self.send_error(404)
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                _zip_workspace_to_file(record.workspace_root, record.job_id, tmp)
                tmp.flush()
                size = tmp.tell()
                tmp.seek(0)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                shutil.copyfileobj(tmp, self.wfile)
        finally:
            self.server.ws._release_download(self.path.strip("/"))

    def do_POST(self):
        token = self.path.strip("/")
        ws = self.server.ws
        record = ws._claim_for_upload(token)
        if record is None:
            return self.send_error(404)
        os.makedirs(record.workspace_root, exist_ok=True)
        try:
            content_length = _parse_content_length(self.headers)
            with _read_request_body_to_tempfile(self.rfile, content_length) as body_file:
                with zipfile.ZipFile(body_file) as zf:
                    _validate_job_zip_members(zf, record.job_id)
                    zf.extractall(record.workspace_root)
        except _RequestError as e:
            ws._release_upload(token, succeeded=False)
            return self.send_error(e.status_code, e.message)
        except (ValueError, zipfile.BadZipFile) as e:
            ws._release_upload(token, succeeded=False)
            return self.send_error(400, str(e))
        ws._release_upload(token, succeeded=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Silence routine 2xx/3xx access logs; keep 4xx/5xx for ops visibility.
        code = args[1] if len(args) > 1 else ""
        if isinstance(code, str) and code[:1] in ("4", "5"):
            logger.warning("ws-http %s - - %s", self.address_string(), fmt % args)


class WorkspaceHTTPServer:
    """Shared HTTPS server for workspace delivery and results collection."""

    def __init__(self, cert_file: str = "", key_file: str = "", require_tls: bool = True):
        self._cert_file = cert_file
        self._key_file = key_file
        self._require_tls = require_tls
        self._port: int = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._scheme = "http"
        self.jobs: dict[str, _JobRecord] = {}  # url_token -> job metadata
        self._in_flight_downloads: set[str] = set()
        self._upload_claimed: set[str] = set()
        self._jobs_lock = threading.Lock()
        self.workspace_root = ""

    @property
    def port(self) -> int:
        return self._port

    def start(self, workspace_root: str, bind: str = "0.0.0.0") -> int:
        self.workspace_root = workspace_root

        if self._require_tls and (not self._cert_file or not self._key_file):
            raise RuntimeError("workspace HTTPS server requires both cert_file and key_file")
        if (self._cert_file or self._key_file) and not (self._cert_file and self._key_file):
            raise RuntimeError("workspace HTTPS server requires both cert_file and key_file")

        srv = ThreadingHTTPServer((bind, 0), _Handler)
        srv.allow_reuse_address = True
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
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("WorkspaceHTTPServer thread is not running; cannot register job")
        token = secrets.token_urlsafe(32)
        with self._jobs_lock:
            self.jobs[token] = _JobRecord(job_id=job_id, workspace_root=workspace_root)
        return token

    def remove_job(self, token: str) -> None:
        with self._jobs_lock:
            self.jobs.pop(token, None)
            self._in_flight_downloads.discard(token)
            self._upload_claimed.discard(token)

    def get_url(self, host: str, token: str) -> str:
        return f"{self._scheme}://{host}:{self._port}/{token}"

    def stop(self) -> None:
        srv = self._server
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    # -- internal, called by the request handler under the serving thread ------
    def _claim_for_download(self, token: str) -> _JobRecord | None:
        with self._jobs_lock:
            record = self.jobs.get(token)
            if record is None:
                return None
            self._in_flight_downloads.add(token)
            return record

    def _release_download(self, token: str) -> None:
        with self._jobs_lock:
            self._in_flight_downloads.discard(token)

    def _claim_for_upload(self, token: str) -> _JobRecord | None:
        with self._jobs_lock:
            record = self.jobs.get(token)
            if record is None or token in self._upload_claimed:
                return None
            self._upload_claimed.add(token)
            return record

    def _release_upload(self, token: str, succeeded: bool) -> None:
        with self._jobs_lock:
            self._upload_claimed.discard(token)
            if succeeded:
                self.jobs.pop(token, None)

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=0.5)
        except Exception:
            logger.exception("workspace HTTP serve loop exited abnormally")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dir_to_zip(zf: zipfile.ZipFile, src: str, root: str) -> None:
    if not os.path.isdir(src):
        return
    for dirpath, _dirs, files in os.walk(src):
        for fname in files:
            abs_path = os.path.join(dirpath, fname)
            zf.write(abs_path, os.path.relpath(abs_path, root))


def _zip_workspace_to_file(workspace_root: str, job_id: str, file_obj) -> None:
    """ZIP local/ and the job directory from *workspace_root* into *file_obj*."""
    with zipfile.ZipFile(file_obj, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_dir_to_zip(zf, os.path.join(workspace_root, "local"), workspace_root)
        _write_dir_to_zip(zf, os.path.join(workspace_root, job_id), workspace_root)


def _zip_workspace(workspace_root: str, job_id: str) -> bytes:
    """ZIP local/ and the job directory from *workspace_root*."""
    buf = io.BytesIO()
    _zip_workspace_to_file(workspace_root, job_id, buf)
    return buf.getvalue()


def _validate_relative_zip_members(zf: zipfile.ZipFile) -> None:
    for info in zf.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe zip member: {name}")


def _validate_job_zip_members(zf: zipfile.ZipFile, job_id: str) -> None:
    _validate_relative_zip_members(zf)
    for info in zf.infolist():
        parts = PurePosixPath(info.filename).parts
        if not parts or parts[0] != job_id:
            raise ValueError(f"zip member outside job workspace: {info.filename}")


def _build_ssl_context(ca_cert: str) -> ssl.SSLContext:
    # URL is a 256-bit capability token; parent is addressed by pod IP which is
    # not in the NVFlare cert SAN. Validate CA chain for encryption/integrity
    # but skip hostname match.
    if not ca_cert or not os.path.exists(ca_cert):
        raise RuntimeError("rootCA.pem is required for workspace HTTPS transfer")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.load_verify_locations(ca_cert)
    return ctx


def _parse_content_length(headers) -> int:
    value = headers.get("Content-Length")
    if value is None:
        raise _RequestError(411, "Content-Length header is required")
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise _RequestError(400, f"invalid Content-Length: {value}")
    if length < 0:
        raise _RequestError(400, "Content-Length must be non-negative")
    if length > MAX_REQUEST_BODY_SIZE:
        raise _RequestError(413, f"request body exceeds {MAX_REQUEST_BODY_SIZE} bytes")
    return length


def _read_request_body_to_tempfile(rfile, content_length: int):
    tmp = tempfile.TemporaryFile()
    remaining = content_length
    while remaining > 0:
        chunk = rfile.read(min(remaining, _STREAM_CHUNK_SIZE))
        if not chunk:
            tmp.close()
            raise _RequestError(400, "incomplete request body")
        tmp.write(chunk)
        remaining -= len(chunk)
    tmp.seek(0)
    return tmp


# ---------------------------------------------------------------------------
# Pod-side functions (read NVFL_WORKSPACE_URL from env; no-op when unset)
# ---------------------------------------------------------------------------


def _retry_transient(op_name: str, fn):
    """Invoke *fn* with bounded exponential backoff on transient transport errors."""
    import time as _time
    import urllib.error

    transient = (urllib.error.URLError, ConnectionError, TimeoutError, ssl.SSLError)
    last_exc: Exception | None = None
    for attempt in range(1, _TRANSFER_RETRIES + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if 500 <= e.code < 600 and attempt < _TRANSFER_RETRIES:
                logger.warning("%s got %s; retrying (attempt %d/%d)", op_name, e.code, attempt, _TRANSFER_RETRIES)
                last_exc = e
            else:
                raise
        except transient as e:
            if attempt < _TRANSFER_RETRIES:
                logger.warning(
                    "%s transient error: %s; retrying (attempt %d/%d)", op_name, e, attempt, _TRANSFER_RETRIES
                )
                last_exc = e
            else:
                raise
        _time.sleep(_TRANSFER_BACKOFF_S * (2 ** (attempt - 1)))
    if last_exc is not None:
        raise last_exc


def download_workspace(dest: str) -> None:
    """GET the workspace ZIP and extract into *dest*. No-op if env var unset."""
    url = os.environ.get(ENV_WORKSPACE_URL)
    if not url:
        return
    import urllib.request

    parsed = urlparse(url)
    ctx = None
    if parsed.scheme == "https":
        ca_cert = os.path.join(dest, "startup", "rootCA.pem")
        ctx = _build_ssl_context(ca_cert)
    os.makedirs(dest, exist_ok=True)
    logger.info("Downloading workspace from %s", url)

    def _attempt():
        with tempfile.TemporaryFile() as tmp:
            with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:
                shutil.copyfileobj(resp, tmp, length=_STREAM_CHUNK_SIZE)
            tmp.seek(0)
            with zipfile.ZipFile(tmp) as zf:
                _validate_relative_zip_members(zf)
                zf.extractall(dest)

    _retry_transient("download_workspace", _attempt)


def upload_results(workspace_root: str, job_id: str) -> None:
    """ZIP the job directory and POST it. No-op if env var unset."""
    url = os.environ.get(ENV_WORKSPACE_URL)
    if not url:
        return
    import urllib.request

    run_dir = os.path.join(workspace_root, job_id)
    if not os.path.isdir(run_dir):
        return

    parsed = urlparse(url)
    ctx = None
    if parsed.scheme == "https":
        ca_cert = os.path.join(workspace_root, "startup", "rootCA.pem")
        ctx = _build_ssl_context(ca_cert)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_dir_to_zip(zf, run_dir, workspace_root)
    data = buf.getvalue()

    def _attempt():
        req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/zip"})
        urllib.request.urlopen(req, timeout=120, context=ctx).close()

    _retry_transient("upload_results", _attempt)
