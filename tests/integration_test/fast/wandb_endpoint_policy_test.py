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

"""Real W&B SDK requests use only the backend approved by the receiving site."""

import base64
import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.process_test_utils import run_owned_process

pytest.importorskip("wandb")


@contextmanager
def listener(key):
    records = []
    expected = {"Basic " + base64.b64encode(f"api:{key}".encode()).decode(), "Bearer " + key}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            records.append(self.headers.get("Authorization") in expected)
            # A deliberate auth failure bounds this local SDK fixture. Online
            # initialization retains the receiver's existing offline fallback.
            body = b'{"errors":[{"message":"fixture unauthorized"}]}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    started = False
    try:
        thread.start()
        started = True
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        if started:
            server.shutdown()
        server.server_close()
        if started:
            thread.join(timeout=10)
            assert not thread.is_alive()


@pytest.mark.timeout(150)
@pytest.mark.parametrize(
    "case", ["default", "approved", "offline", "default-run", "approved-run", "reject-online", "reject-run"]
)
def test_real_sdk_backend_approval_and_offline_logging(tmp_path, case):
    key = "a" * 40
    with listener(key) as (trusted, trusted_requests), listener(key) as (selected, selected_requests):
        config = {
            "workspace": str(tmp_path / "workspace"),
            "result": str(tmp_path / "result.json"),
            "mode": "offline" if case == "offline" else ("run" if case.endswith("-run") else "online"),
        }
        if case in ("approved", "approved-run", "reject-online", "reject-run"):
            config["selected"] = selected
        if case in ("approved", "approved-run"):
            config["approved"] = [selected]
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_") and not k.lower().endswith("_proxy")}
        env.update(
            WANDB_API_KEY=key,
            WANDB_BASE_URL=trusted,
            WANDB_ERROR_REPORTING="false",
            WANDB_CONFIG_DIR=str(tmp_path / "sdk-config"),
            WANDB_CACHE_DIR=str(tmp_path / "cache"),
            NETRC=str(tmp_path / "absent.netrc"),
            PYTHONPATH=str(Path(__file__).resolve().parents[3]),
        )
        result = run_owned_process(
            [sys.executable, "-m", "tests.wandb_receiver_test_utils", str(config_path)],
            cwd=tmp_path,
            env=env,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads((tmp_path / "result.json").read_text())
        print(json.dumps({"trusted_auth_matches": trusted_requests, "selected_auth_matches": selected_requests}))
        if case.startswith("reject"):
            assert report["status"] == "blocked", (
                report,
                {"trusted": trusted_requests, "selected": selected_requests},
                result.stderr,
            )
            assert report["workers_started"] == 0
            assert trusted_requests == selected_requests == []
        elif case in ("default-run", "approved-run"):
            # No login preflight: the actual queued wandb.init call sends its
            # credential only to the approved backend, then receives our 401.
            assert report["status"] == "sdk_error", (report, result.stderr)
            assert report["error_type"] == "CommError"
            assert report["workers_started"] == 1
            if case == "default-run":
                assert trusted_requests and all(trusted_requests)
                assert selected_requests == []
            else:
                # Older SDKs verify environment keys against the site default
                # before using a per-run host. Both destinations are approved.
                assert trusted_requests or selected_requests
                assert all(trusted_requests + selected_requests)
        else:
            assert report["status"] == "finished", (report, result.stderr)
            assert report["workers_started"] == 1
            # dir defaults to the child working directory, outside workspace.
            assert list(tmp_path.rglob("*.wandb"))
            if case == "default":
                assert trusted_requests and all(trusted_requests)
                assert selected_requests == []
            elif case == "approved":
                assert trusted_requests or selected_requests
                assert all(trusted_requests + selected_requests)
            else:
                assert trusted_requests == selected_requests == []
