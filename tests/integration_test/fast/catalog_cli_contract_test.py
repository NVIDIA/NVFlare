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
if sys.argv.pop(1) == "with-entries":
    from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ColumnInfo, Provenance, TableInfo
    from nvflare.app_common.fdn.catalog.server.commands import SEARCH
    from nvflare.fuel.hci.proto import ProtoKey
    catalog = CatalogInfo("omop", "OMOP", Provenance("sql"), tables=(TableInfo("person", (ColumnInfo("id", "INTEGER"),)),))
    original_command = session.do_command
    def command_with_entries(command, props):
        reply = original_command(command, props)
        if command == SEARCH:
            payload = reply[ProtoKey.DATA][0][ProtoKey.DATA]
            payload["entries"] = [dict(client=site, dataset=dataset, available=True, last_checked_at=None,
                metadata=catalog.metadata_wire(), catalog_version=1, schema_version=1, published_at=None)
                for site, dataset in (("site-a", "omop"), ("site-a", "cohort"), ("site-b", "omop"))]
            payload["total"] = len(payload["entries"])
        return reply
    session.do_command = command_with_entries
cli_session.new_cli_session_for_args = lambda **kw: session
if "--schema" in sys.argv:
    def unexpected_connection(**kw):
        raise AssertionError("schema discovery must not connect")
    cli_session.new_cli_session_for_args = unexpected_connection
sys.argv[0] = "nvflare"
cli.main()
"""


def invoke(*args, status="FINISHED:COMPLETED", with_entries=False):
    # Completion watchdog permits interpreter startup and CI contention; it is not a product timeout.
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, status, "with-entries" if with_entries else "empty", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
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
    else:
        args = {arg["name"]: arg for arg in schema["args"]}
        assert args["--overview"]["default"] is False


@pytest.mark.parametrize("overview", [False, True])
def test_complete_json_export_and_overview_are_distinct(overview):
    args = ["catalog", "show", "--format", "json"] + (["--overview"] if overview else [])
    proc = invoke(*args, with_entries=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)["data"]
    assert data["total"] == 3
    assert [(d["site"], d["dataset"]) for d in data["datasets"]] == [
        ("site-a", "omop"),
        ("site-a", "cohort"),
        ("site-b", "omop"),
    ]
    for detail in data["datasets"]:
        assert ("metadata" in detail) is not overview
        if not overview:
            assert detail["metadata"]["tables"][0]["columns"][0]["name"] == "id"
    assert "incomplete" in proc.stderr


@pytest.mark.parametrize("args", [("--sites", ","), ("--timeout", "0"), ("bad",), ("--unknown",), ("--wait",)])
def test_bad_arguments_have_one_error_envelope_and_exit_four(args):
    proc = invoke("catalog", "read", *args, "--format", "json")
    assert proc.returncode == 4, (proc.stdout, proc.stderr)
    assert json.loads(proc.stdout)["status"] == "error"
