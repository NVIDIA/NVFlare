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

"""Failure summaries preserve actionable diagnostics without scanning whole logs."""

import json
import logging
import os
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.utils.log_utils import BaseFormatter
from nvflare.recipe._failure_summary import collect_client_errors, failure_summary
from nvflare.recipe.run import Run


def _record(message, name="TaskScriptRunner", context=""):
    return json.dumps({"levelname": "ERROR", "name": name, "message": message, "fl_ctx": context}) + "\n"


def _client_trace(site, error="FileNotFoundError: missing training.npy"):
    return (
        "Traceback (most recent call last):\n"
        f'  File "/workspace/{site}/custom/client.py", line 31, in train\n'
        '    np.load("training.npy")\n'
        '  File "/packages/numpy/io.py", line 90, in load\n'
        "    open(path)\n"
        f"{error}\n"
    )


def _write_log(root, site, text, plain_text=False):
    folder = root / site
    folder.mkdir(parents=True, exist_ok=True)
    if plain_text:
        formatter = BaseFormatter("%(asctime)s - %(name)s - %(levelname)s - %(fl_ctx)s - %(message)s")
        text = "\n".join(
            formatter.format(logging.makeLogRecord({**json.loads(line), "msg": json.loads(line)["message"]}))
            for line in text.splitlines()
        )
    else:
        (folder / "log.json").write_text(text)
    (folder / "error_log.txt").write_text(text)


@pytest.mark.parametrize("layout", [".", "workspace"])
@pytest.mark.parametrize("plain_text", [False, True])
@pytest.mark.parametrize("workflow", ["train", "train[0]"])
def test_groups_client_errors_and_omits_only_the_linked_abort(tmp_path, layout, plain_text, workflow):
    root = tmp_path / layout
    for site in ("site-1", "site-2"):
        _write_log(root, site, _record(_client_trace(site)) + _record("fire abort event"), plain_text)
    _write_log(
        root,
        "server",
        _record(
            "downstream invalid DXO", context=f"[wf={workflow}, peer=site-1, task={workflow}, peer_rc=TASK_ABORTED]"
        ),
        plain_text,
    )
    output = failure_summary(tmp_path)
    assert output.count("FileNotFoundError: missing training.npy") == 1
    assert "site-1, site-2" in output
    assert "client.py:31 (train)" in output
    assert "site-1/error_log.txt" in output
    assert "downstream invalid DXO" not in output
    assert "fire abort event" not in output
    assert max(map(len, output.splitlines())) <= 80


@pytest.mark.parametrize("plain_text", [False, True])
def test_unrelated_client_error_does_not_hide_another_peers_abort(tmp_path, plain_text):
    _write_log(tmp_path, "site-1", _record(_client_trace("site-1")), plain_text)
    _write_log(
        tmp_path, "server", _record("site-2 task aborted", context="[peer=site-2, peer_rc=TASK_ABORTED]"), plain_text
    )
    output = failure_summary(tmp_path)
    assert "site-2 task aborted" in output
    assert "Also reported" in output


def test_server_only_download_is_explicit_about_missing_client_logs(tmp_path):
    _write_log(tmp_path, "workspace", _record("ConnectionError: server connection lost", name="Cell"))
    output = failure_summary(tmp_path)
    assert "ConnectionError: server connection lost" in output
    assert "server / Cell" in output
    assert "Client logs are not included" in output


def test_malformed_and_oversized_logs_do_not_hide_recent_errors(tmp_path):
    _write_log(
        tmp_path,
        "site-1",
        "X" * (2 * 1024 * 1024) + "\n{bad json}\n[]\n" + _record(_client_trace("site-1")),
    )
    output = failure_summary(tmp_path)
    assert "FileNotFoundError" in output
    assert len(output) < 1500


def test_no_logs_and_stale_simulator_logs_have_honest_fallback(tmp_path):
    assert "No job error details" in failure_summary(tmp_path)
    _write_log(tmp_path, "site-1", _record("NameError: old job"))
    os.utime(tmp_path / "site-1" / "log.json", (1, 1))
    output = failure_summary(tmp_path, since=2)
    assert "old job" not in output
    assert "No job error details" in output


def test_summary_does_not_follow_log_symlinks_outside_result(tmp_path):
    external = tmp_path / "external.json"
    external.write_text(_record("outside result"))
    result = tmp_path / "result"
    result.mkdir()
    (result / "log.json").symlink_to(external)
    assert "outside result" not in failure_summary(result)


@pytest.mark.parametrize(
    "status", ["FINISHED:EXECUTION_EXCEPTION", "FAILED", "FINISHED_EXCEPTION", "ABORTED", "ABANDONED"]
)
def test_failed_run_summarizes_before_cleanup_and_only_once(tmp_path, capsys, status):
    _write_log(tmp_path, "workspace", _record("OSError: checkpoint write failed", name="ModelPersistor"))
    env = MagicMock()
    env.get_job_result.return_value = str(tmp_path)
    env.get_job_status.return_value = status
    env.stop.side_effect = lambda **kwargs: (tmp_path / "workspace" / "log.json").unlink()
    run = Run(env, "failed-job")
    assert run.get_result() == str(tmp_path)
    output = capsys.readouterr().out
    assert "RUN SUMMARY" in output
    assert "OSError: checkpoint write failed" in output
    assert "✓ Completed" not in output
    assert "✗ Failed" in output
    assert run.get_status() == status
    run.get_result()
    assert capsys.readouterr().out == ""


def test_existing_client_error_streams_are_saved_and_summarized(tmp_path):
    session = MagicMock()
    session.get_job_logs.return_value = {
        "logs": {
            "server": "ignored duplicate server log",
            "../outside": "must not be written",
            **{
                site: "2026-09-12 10:40:51,611 - TaskScriptRunner - ERROR - "
                + _client_trace(site)
                + "2026-09-12 10:40:51,612 - TaskScriptRunner - ERROR - fire abort event\n"
                for site in ("site-1", "site-2")
            },
        },
        "unavailable": {"site-3": "streaming disabled"},
    }
    response = session.get_job_logs.return_value
    session.list_job_components.return_value = ["workspace", *[f"ERRORLOG_{site}" for site in response["logs"]]]
    session.get_job_logs.side_effect = lambda job_id, target, **kwargs: {"logs": {target: response["logs"][target]}}
    collect_client_errors(session, "job-id", str(tmp_path))
    assert session.get_job_logs.call_count == 2
    assert {call.kwargs["target"] for call in session.get_job_logs.call_args_list} == {"site-1", "site-2"}
    assert all(call.kwargs["max_bytes"] == 1024 * 1024 for call in session.get_job_logs.call_args_list)
    output = failure_summary(tmp_path)
    assert output.count("FileNotFoundError: missing training.npy") == 1
    assert "site-1, site-2 / TaskScriptRunner" in output
    assert "client.py:31 (train)" in output
    assert "fire abort event" not in output
    assert len(list(tmp_path.rglob("error_log.txt"))) == 2
    assert "site-3" not in output  # Never invent a diagnosis for an unavailable site.


def test_client_error_download_is_bounded(tmp_path):
    session = MagicMock()
    session.get_job_logs.return_value = {
        "logs": {f"site-{n}": "X" * (1024 * 1024) + "\nlast line\n" for n in range(25)}
    }
    session.list_job_components.return_value = [f"ERRORLOG_site-{n}" for n in range(25)]
    collect_client_errors(session, "job-id", str(tmp_path))
    paths = list(tmp_path.rglob("error_log.txt"))
    assert len(paths) == 20
    assert session.get_job_logs.call_count == 20
    assert all(call.kwargs["target"] != "all" for call in session.get_job_logs.call_args_list)
    assert all(call.kwargs["max_bytes"] == 1024 * 1024 for call in session.get_job_logs.call_args_list)
    assert all(path.stat().st_size <= 1024 * 1024 for path in paths)
    assert all(path.read_text() == "last line\n" for path in paths)
