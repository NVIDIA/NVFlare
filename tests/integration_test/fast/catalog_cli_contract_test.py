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

"""Real CLI parsing, recipe export, stream routing and process exits; only the session is fake."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = """
import sys
from nvflare import cli
from nvflare.tool import cli_session
from tests.catalog_cli_test_utils import CatalogSession
session = CatalogSession(status=sys.argv.pop(1), health="DEGRADED")
cli_session.new_cli_session_for_args = lambda **kw: session
if "--schema" in sys.argv:
    def unexpected_connection(**kw):
        raise AssertionError("schema discovery must not connect")
    cli_session.new_cli_session_for_args = unexpected_connection
sys.argv[0] = "nvflare"
cli.main()
"""


def invoke(*args, status="FINISHED:COMPLETED"):
    # Completion watchdog permits interpreter startup and CI contention; it is not a product timeout.
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, status, *args], cwd=ROOT, capture_output=True, text=True, timeout=90
    )


@pytest.mark.parametrize(
    "status,code,error",
    [
        ("FINISHED:COMPLETED", 0, None),
        ("FINISHED:EXECUTION_EXCEPTION", 1, "JOB_FAILED"),
        ("FINISHED:ABORTED", 1, "JOB_ABORTED"),
        ("FINISHED:CAN_NOT_SCHEDULE", 1, "JOB_FAILED"),
        ("TIMEOUT", 3, "TIMEOUT"),
    ],
)
def test_default_wait_reports_job_outcome_as_one_json_envelope(status, code, error):
    proc = invoke("catalog", "read", "--study", "oncology", "--format", "json", status=status)
    assert proc.returncode == code, (proc.stdout, proc.stderr)
    envelope = json.loads(proc.stdout)  # Rejects banners or multiple JSON documents on stdout.
    assert envelope["schema_version"] == "1" and envelope["exit_code"] == code
    assert envelope["status"] == ("error" if error else "ok")
    if error:
        assert envelope["error_code"] == error
    else:
        assert envelope["data"] == {"study": "oncology", "job_id": "job-1", "status": status}
    assert "waiting for completion" in proc.stderr and "RUNNING" in proc.stderr


def test_no_wait_json_reports_submission_and_human_output_is_readable():
    proc = invoke("--format", "json", "catalog", "read", "--no-wait")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["data"] == {"study": "default", "job_id": "job-1"}
    assert "waiting for completion" not in proc.stderr
    human = invoke("catalog", "read")
    assert human.returncode == 0, human.stderr
    assert "finished: FINISHED:COMPLETED" in human.stdout and "catalog show" in human.stdout


@pytest.mark.parametrize("fmt", ["txt", "json"])
def test_show_exposes_degraded_sites_even_with_no_datasets(fmt):
    proc = invoke("catalog", "show", "--format", fmt)
    assert proc.returncode == 0, proc.stderr
    if fmt == "json":
        assert json.loads(proc.stdout)["data"]["sites"][0]["health"] == "DEGRADED"
        diagnostic = proc.stderr
    else:
        assert "no datasets" in proc.stdout
        diagnostic = proc.stdout
    assert "incomplete (BUDGET_EXCEEDED)" in diagnostic and "Select fewer datasets" in diagnostic


@pytest.mark.parametrize("command,mutating", [("show", False), ("read", True)])
def test_schema_needs_no_kit_and_describes_machine_contract(command, mutating):
    proc = invoke("catalog", command, "--schema")
    assert proc.returncode == 0, proc.stderr
    schema = json.loads(proc.stdout)
    assert schema["command"] == f"nvflare catalog {command}"
    assert schema["output_modes"] == ["json"] and schema["streaming"] is False
    assert schema["mutating"] is mutating and schema["idempotent"] is not mutating
    assert schema["retry_token"] == {"supported": False}
    if command == "read":
        args = {arg["name"]: arg for arg in schema["args"]}
        assert args["--no-wait"]["default"] is False
        assert "--wait" not in args


@pytest.mark.parametrize("args", [("--sites", ","), ("--timeout", "0"), ("--only", "bad"), ("--unknown",), ("--wait",)])
def test_bad_arguments_have_one_error_envelope_and_exit_four(args):
    proc = invoke("catalog", "read", *args, "--format", "json")
    assert proc.returncode == 4, (proc.stdout, proc.stderr)
    assert json.loads(proc.stdout)["status"] == "error"
