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

from unittest.mock import Mock

import pytest

from tests.integration_test.slow import distributed_provisioning_test as distributed
from tests.integration_test.src import utils
from tests.process_test_utils import OwnedProcesses
from tests.timing_utils import ManualClock


@pytest.mark.parametrize("failure", ["client_launch", "admin_connect", "test_body"])
def test_failed_setup_or_body_stops_every_started_site(tmp_path, monkeypatch, failure):
    ca = tmp_path / "ca"
    ca.mkdir()
    (ca / "rootCA.pem").write_bytes(b"controlled CA")
    monkeypatch.setattr(distributed, "Project", Mock())
    monkeypatch.setattr(distributed, "load_private_key_file", Mock())
    monkeypatch.setattr(
        distributed,
        "Provisioner",
        Mock(return_value=Mock(provision=Mock(return_value={"current_prod_dir": str(tmp_path)}))),
    )
    clock = ManualClock()
    clock.sleep = clock.advance
    monkeypatch.setattr(distributed, "time", clock)
    monkeypatch.setattr(distributed, "_STARTUP_WAIT", 0)
    monkeypatch.setattr(distributed, "_ADMIN_CONNECT_TIMEOUT", 0.2)
    monkeypatch.setattr(OwnedProcesses, "capture", lambda *_, **__: None)
    started = []

    def launch(*args, **kwargs):
        if failure == "client_launch" and started:
            raise OSError("controlled client launch failure")
        process = Mock(pid=100 + len(started))
        process.poll.return_value = None
        process.terminate.side_effect = lambda: setattr(process.poll, "return_value", 0)
        started.append(process)
        return process

    monkeypatch.setattr(distributed, "popen_in_new_session", launch)
    api = Mock()
    connect = Mock(
        return_value=api, side_effect=ConnectionError("connection failure") if failure != "test_body" else None
    )
    monkeypatch.setattr(utils, "create_admin_api", connect)
    kits = {name: str(tmp_path / name) for name in ("site-1", "site-2", "admin@myfl.com")}
    fixture = distributed.TestDistributedProvisioningE2E.admin_api.__wrapped__(
        distributed.TestDistributedProvisioningE2E(), kits, tmp_path
    )
    if failure == "test_body":
        assert next(fixture) is api
        with pytest.raises(ValueError, match="body failed"):
            fixture.throw(ValueError("body failed"))
        api.close.assert_called_once()
    else:
        with pytest.raises((OSError, pytest.fail.Exception)):
            next(fixture)
    assert len(started) == (1 if failure == "client_launch" else 3)
    for process in started:
        process.terminate.assert_called_once()
        assert process.wait.called


def test_external_timeout_reports_original_output_logs_and_cleanup_failure(tmp_path, monkeypatch):
    import subprocess

    from tests.integration_test.slow import external_process_e2e_test as external

    (tmp_path / "log.txt").write_text("last simulator activity")
    error = subprocess.TimeoutExpired("original command", 300, output="original output", stderr="original stderr")
    error.cleanup_error = "surviving trainer pid=1234"
    monkeypatch.setattr(external, "run_owned_process", Mock(side_effect=error))
    with pytest.raises(pytest.fail.Exception) as caught:
        external._run_job_with_timeout_diagnostics(["job"], str(tmp_path), {}, str(tmp_path), 300)
    message = str(caught.value)
    assert "timed out after 300" in message
    assert "original outputoriginal stderr" in message
    assert "last simulator activity" in message
    assert "surviving trainer pid=1234" in message
