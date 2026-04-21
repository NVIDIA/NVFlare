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

import http.client
import io
import os
import tempfile
import urllib.error
import urllib.request
import zipfile

import pytest

from nvflare.app_opt.job_launcher.workspace_http_server import (
    ENV_WORKSPACE_URL,
    MAX_REQUEST_BODY_SIZE,
    WorkspaceHTTPServer,
    _zip_workspace,
    download_workspace,
    upload_results,
)

JOB_ID = "abc12345-dead-beef-0000-111122223333"
JOB_ID_B = "bbbbbbbb-dead-beef-0000-111122223333"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: str, content: bytes = b"data") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)


def _make_workspace(root: str, job_id: str) -> dict:
    """Populate a minimal workspace; return {rel_path: bytes} for run_<job_id>/ files."""
    _write_file(os.path.join(root, "startup", "fed_client.json"), b'{"type":"client"}')
    local_files = {
        os.path.join("local", "resources.json"): b'{"resources":{}}',
        os.path.join("local", "job_resources.json"): b'{"job_resources":{}}',
        os.path.join("local", "site__j_resources.json"): b'{"job_only": true}',
        os.path.join("local", "custom", "site_module.py"): b"VALUE = 3\n",
    }
    run_files = {os.path.join(job_id, "app", "config", "config_train.json"): b'{"rounds":3}'}
    expected = {}
    expected.update(local_files)
    expected.update(run_files)
    for rel, content in local_files.items():
        _write_file(os.path.join(root, rel), content)
    for rel, content in run_files.items():
        _write_file(os.path.join(root, rel), content)
    return expected


def _make_result_files(root: str, job_id: str) -> dict:
    files = {
        os.path.join(job_id, "fl_app.txt"): b"done",
        os.path.join(job_id, "app_server", "best_model.pt"): b"WEIGHTS",
        os.path.join(job_id, "log.txt"): b"training log\n",
    }
    for rel, content in files.items():
        _write_file(os.path.join(root, rel), content)
    return files


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _start_server(ws_root: str, job_id: str = JOB_ID):
    """Start a shared server, register one job, return (server, url_token)."""
    server = WorkspaceHTTPServer(require_tls=False)
    server.start(workspace_root=ws_root)
    url_token = server.add_job(job_id, ws_root)
    return server, url_token


# ---------------------------------------------------------------------------
# TestZipWorkspace
# ---------------------------------------------------------------------------


class TestZipWorkspace:
    def test_result_is_valid_zip(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            data = _zip_workspace(ws_root, JOB_ID)
            assert zipfile.is_zipfile(io.BytesIO(data))

    def test_zip_does_not_contain_startup_files(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            data = _zip_workspace(ws_root, JOB_ID)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
            assert not any(n.startswith("startup") for n in names)

    def test_zip_contains_full_local_tree(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            data = _zip_workspace(ws_root, JOB_ID)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
            assert "local/job_resources.json" in names
            assert "local/site__j_resources.json" in names
            assert "local/custom/site_module.py" in names

    def test_zip_contains_run_dir_files(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            data = _zip_workspace(ws_root, JOB_ID)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
            assert any(n.startswith(JOB_ID) for n in names)

    def test_missing_run_dir_returns_empty_zip(self):
        with tempfile.TemporaryDirectory() as ws_root:
            data = _zip_workspace(ws_root, JOB_ID)
            assert zipfile.is_zipfile(io.BytesIO(data))
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                assert zf.namelist() == []

    def test_arcnames_are_relative(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            data = _zip_workspace(ws_root, JOB_ID)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    assert not os.path.isabs(name)

    def test_file_contents_preserved(self):
        with tempfile.TemporaryDirectory() as ws_root:
            expected = b"expected content"
            _write_file(os.path.join(ws_root, JOB_ID, "sentinel.bin"), expected)
            data = _zip_workspace(ws_root, JOB_ID)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                name = next(n for n in zf.namelist() if "sentinel.bin" in n)
                assert zf.read(name) == expected

    def test_different_job_ids_produce_different_archives(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _write_file(os.path.join(ws_root, JOB_ID, "a.txt"), b"a")
            _write_file(os.path.join(ws_root, JOB_ID_B, "b.txt"), b"b")
            data_a = _zip_workspace(ws_root, JOB_ID)
            data_b = _zip_workspace(ws_root, JOB_ID_B)
            with zipfile.ZipFile(io.BytesIO(data_a)) as zf:
                assert not any(JOB_ID_B in n for n in zf.namelist())
            with zipfile.ZipFile(io.BytesIO(data_b)) as zf:
                assert not any(JOB_ID in n for n in zf.namelist())


# ---------------------------------------------------------------------------
# TestWorkspaceHTTPServer
# ---------------------------------------------------------------------------


class TestWorkspaceHTTPServer:
    def test_start_requires_tls_by_default(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer()
            with pytest.raises(RuntimeError, match="cert_file"):
                server.start(workspace_root=ws_root)

    def test_start_returns_nonzero_port(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            try:
                port = server.start(workspace_root=ws_root)
                assert isinstance(port, int) and port > 0
            finally:
                server.stop()

    def test_port_property_matches_start(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            try:
                port = server.start(workspace_root=ws_root)
                assert server.port == port
            finally:
                server.stop()

    def test_add_job_returns_url_token(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            try:
                url_token = server.add_job(JOB_ID, ws_root)
                assert isinstance(url_token, str) and len(url_token) > 0
            finally:
                server.stop()

    def test_each_job_gets_unique_token(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            try:
                token_a = server.add_job(JOB_ID, ws_root)
                token_b = server.add_job(JOB_ID_B, ws_root)
                assert token_a != token_b
            finally:
                server.stop()

    def test_get_url_includes_token_in_path(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            try:
                port = server.start(workspace_root=ws_root)
                url_token = server.add_job(JOB_ID, ws_root)
                url = server.get_url("10.0.0.5", url_token)
                assert url == f"http://10.0.0.5:{port}/{url_token}"
            finally:
                server.stop()

    def test_get_with_valid_token_returns_zip(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            server, url_token = _start_server(ws_root)
            try:
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    assert resp.status == 200
                    body = resp.read()
                assert zipfile.is_zipfile(io.BytesIO(body))
            finally:
                server.stop()

    def test_get_zips_workspace_at_request_time(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            try:
                _write_file(os.path.join(ws_root, JOB_ID, "late.txt"), b"late")
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = resp.read()
                with zipfile.ZipFile(io.BytesIO(body)) as zf:
                    assert zf.read(f"{JOB_ID}/late.txt") == b"late"
            finally:
                server.stop()

    def test_unknown_token_returns_404(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            try:
                url = f"http://127.0.0.1:{server.port}/no-such-token"
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(url, timeout=10)
                assert exc.value.code == 404
            finally:
                server.stop()

    def test_two_jobs_are_served_independently(self):
        with tempfile.TemporaryDirectory() as ws_root:
            _write_file(os.path.join(ws_root, JOB_ID, "a.txt"), b"aaa")
            _write_file(os.path.join(ws_root, JOB_ID_B, "b.txt"), b"bbb")
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            try:
                token_a = server.add_job(JOB_ID, ws_root)
                token_b = server.add_job(JOB_ID_B, ws_root)

                with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/{token_a}", timeout=10) as r:
                    zip_a = r.read()
                with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/{token_b}", timeout=10) as r:
                    zip_b = r.read()

                with zipfile.ZipFile(io.BytesIO(zip_a)) as zf:
                    assert any(JOB_ID in n for n in zf.namelist())
                    assert not any(JOB_ID_B in n for n in zf.namelist())

                with zipfile.ZipFile(io.BytesIO(zip_b)) as zf:
                    assert any(JOB_ID_B in n for n in zf.namelist())
                    assert not any(JOB_ID in n for n in zf.namelist())
            finally:
                server.stop()

    def test_upload_rejects_members_outside_job_dir(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            try:
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                data = _make_zip({"startup/pwned.txt": b"nope"})
                req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/zip"})
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(req, timeout=10)
                assert exc.value.code == 400
                assert not os.path.exists(os.path.join(ws_root, "startup", "pwned.txt"))
                assert url_token in server.jobs
            finally:
                server.stop()

    def test_upload_rejects_other_job_dir(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            try:
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                data = _make_zip({f"{JOB_ID_B}/pwned.txt": b"nope"})
                req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/zip"})
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(req, timeout=10)
                assert exc.value.code == 400
                assert not os.path.exists(os.path.join(ws_root, JOB_ID_B, "pwned.txt"))
                assert url_token in server.jobs
            finally:
                server.stop()

    def test_upload_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            try:
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                data = _make_zip({f"{JOB_ID}/../pwned.txt": b"nope"})
                req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/zip"})
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(req, timeout=10)
                assert exc.value.code == 400
                assert not os.path.exists(os.path.join(ws_root, "pwned.txt"))
                assert url_token in server.jobs
            finally:
                server.stop()

    def test_remove_job_makes_subsequent_requests_return_404(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            try:
                server.remove_job(url_token)
                url = f"http://127.0.0.1:{server.port}/{url_token}"
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(url, timeout=10)
                assert exc.value.code == 404
            finally:
                server.stop()

    def test_post_requires_content_length(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
            try:
                conn.putrequest("POST", f"/{url_token}")
                conn.endheaders()
                resp = conn.getresponse()
                assert resp.status == 411
                resp.read()
            finally:
                conn.close()
                server.stop()

    def test_post_rejects_invalid_content_length(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
            try:
                conn.putrequest("POST", f"/{url_token}")
                conn.putheader("Content-Length", "bogus")
                conn.endheaders()
                resp = conn.getresponse()
                assert resp.status == 400
                resp.read()
            finally:
                conn.close()
                server.stop()

    def test_post_rejects_oversized_content_length(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server, url_token = _start_server(ws_root)
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
            try:
                conn.putrequest("POST", f"/{url_token}")
                conn.putheader("Content-Length", str(MAX_REQUEST_BODY_SIZE + 1))
                conn.endheaders()
                resp = conn.getresponse()
                assert resp.status == 413
                resp.read()
            finally:
                conn.close()
                server.stop()

    def test_stop_terminates_server_thread(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            server.stop()
            assert not server._thread.is_alive()

    def test_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as ws_root:
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            server.stop()
            server.stop()  # must not raise


# ---------------------------------------------------------------------------
# TestDownloadWorkspace
# ---------------------------------------------------------------------------


class TestDownloadWorkspace:
    def test_noop_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_WORKSPACE_URL, raising=False)
        with tempfile.TemporaryDirectory() as dest:
            download_workspace(dest)  # must not raise

    def test_end_to_end_files_extracted(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as dest:
            expected = _make_workspace(ws_root, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                download_workspace(dest)
            finally:
                server.stop()
            for rel in expected:
                assert os.path.exists(os.path.join(dest, rel))

    def test_file_contents_match(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as dest:
            expected = _make_workspace(ws_root, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                download_workspace(dest)
            finally:
                server.stop()
            for rel, content in expected.items():
                with open(os.path.join(dest, rel), "rb") as fh:
                    assert fh.read() == content

    def test_creates_dest_if_missing(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _make_workspace(ws_root, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            dest = os.path.join(tmp, "nonexistent", "subdir")
            try:
                download_workspace(dest)
            finally:
                server.stop()
            assert os.path.isdir(dest)

    def test_wrong_url_raises_404(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as dest:
            server, _ = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, f"http://127.0.0.1:{server.port}/wrong-token")
            try:
                with pytest.raises(urllib.error.HTTPError) as exc:
                    download_workspace(dest)
                assert exc.value.code == 404
            finally:
                server.stop()

    def test_https_requires_root_ca(self, monkeypatch):
        monkeypatch.setenv(ENV_WORKSPACE_URL, "https://example.invalid/token")
        with tempfile.TemporaryDirectory() as dest:
            with pytest.raises(RuntimeError, match="rootCA.pem"):
                download_workspace(dest)


# ---------------------------------------------------------------------------
# TestResultsUpload
# ---------------------------------------------------------------------------


class TestResultsUpload:
    def test_noop_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_WORKSPACE_URL, raising=False)
        with tempfile.TemporaryDirectory() as pod_ws:
            upload_results(pod_ws, JOB_ID)  # must not raise

    def test_full_round_trip(self, monkeypatch):
        with (
            tempfile.TemporaryDirectory() as ws_root,
            tempfile.TemporaryDirectory() as pod_ws,
        ):
            _make_workspace(ws_root, JOB_ID)
            result_files = _make_result_files(pod_ws, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                upload_results(pod_ws, JOB_ID)
            finally:
                server.stop()
            for rel, content in result_files.items():
                extracted = os.path.join(ws_root, rel)
                assert os.path.exists(extracted)
                with open(extracted, "rb") as fh:
                    assert fh.read() == content

    def test_wrong_url_returns_404(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as pod_ws:
            _make_result_files(pod_ws, JOB_ID)
            server, _ = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, f"http://127.0.0.1:{server.port}/wrong-token")
            try:
                with pytest.raises(urllib.error.HTTPError) as exc:
                    upload_results(pod_ws, JOB_ID)
                assert exc.value.code == 404
            finally:
                server.stop()

    def test_missing_run_dir_does_not_raise(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as empty_pod_ws:
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                upload_results(empty_pod_ws, JOB_ID)
            finally:
                server.stop()

    def test_job_auto_removed_from_registry_after_upload(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as pod_ws:
            _make_result_files(pod_ws, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                upload_results(pod_ws, JOB_ID)
                assert url_token not in server.jobs
            finally:
                server.stop()

    def test_https_requires_root_ca(self, monkeypatch):
        with tempfile.TemporaryDirectory() as pod_ws:
            _make_result_files(pod_ws, JOB_ID)
            monkeypatch.setenv(ENV_WORKSPACE_URL, "https://example.invalid/token")
            with pytest.raises(RuntimeError, match="rootCA.pem"):
                upload_results(pod_ws, JOB_ID)


# ---------------------------------------------------------------------------
# TestFullEndToEnd
# ---------------------------------------------------------------------------


class TestFullEndToEnd:
    def test_download_then_upload(self, monkeypatch):
        with (
            tempfile.TemporaryDirectory() as ws_root,
            tempfile.TemporaryDirectory() as pod_ws,
        ):
            _make_workspace(ws_root, JOB_ID)
            result_files = _make_result_files(pod_ws, JOB_ID)
            server, url_token = _start_server(ws_root)
            monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", url_token))
            try:
                download_workspace(pod_ws)
                upload_results(pod_ws, JOB_ID)
            finally:
                server.stop()
            for rel, content in result_files.items():
                with open(os.path.join(ws_root, rel), "rb") as fh:
                    assert fh.read() == content

    def test_two_parallel_jobs_full_round_trip(self, monkeypatch):
        with (
            tempfile.TemporaryDirectory() as ws_root,
            tempfile.TemporaryDirectory() as pod_ws_a,
            tempfile.TemporaryDirectory() as pod_ws_b,
        ):
            _write_file(os.path.join(ws_root, JOB_ID, "a.txt"), b"aaa")
            _write_file(os.path.join(ws_root, JOB_ID_B, "b.txt"), b"bbb")
            server = WorkspaceHTTPServer(require_tls=False)
            server.start(workspace_root=ws_root)
            token_a = server.add_job(JOB_ID, ws_root)
            token_b = server.add_job(JOB_ID_B, ws_root)
            try:
                monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", token_a))
                download_workspace(pod_ws_a)

                monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", token_b))
                download_workspace(pod_ws_b)

                monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", token_a))
                _make_result_files(pod_ws_a, JOB_ID)
                upload_results(pod_ws_a, JOB_ID)

                monkeypatch.setenv(ENV_WORKSPACE_URL, server.get_url("127.0.0.1", token_b))
                _make_result_files(pod_ws_b, JOB_ID_B)
                upload_results(pod_ws_b, JOB_ID_B)

                assert token_a not in server.jobs
                assert token_b not in server.jobs
            finally:
                server.stop()


# ---------------------------------------------------------------------------
# TestEnvConstant
# ---------------------------------------------------------------------------


class TestEnvConstant:
    def test_env_workspace_url(self):
        assert ENV_WORKSPACE_URL == "NVFL_WORKSPACE_URL"
