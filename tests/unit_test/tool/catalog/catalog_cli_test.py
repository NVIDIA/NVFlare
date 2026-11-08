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

import argparse
import json
from types import SimpleNamespace

import pytest

from nvflare.app_common.fdn.catalog.admin.catalog_api import ClientHealth
from nvflare.app_common.fdn.catalog.server.search import CatalogEntry
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    CommandError,
    InternalError,
    InvalidArgumentError,
    InvalidJobDefinition,
    JobNotFound,
    NoConnection,
    SessionExpired,
)
from nvflare.tool import cli_output, cli_session
from nvflare.tool.catalog import catalog_cli
from tests.catalog_cli_test_utils import CatalogSession
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog


def health(client="site-a", **kw):
    row = dict(
        client=client,
        health="OK",
        reason_code=None,
        failures=0,
        last_attempt_at="2026-06-01T00:00:20Z",
        last_success_at="2026-06-01T00:00:20Z",
        stale=False,
    )
    row.update(kw)
    return ClientHealth(**row)


class FakeClient:
    def __init__(self, statuses=(), entries=()):
        self.statuses, self.entries, self.filters = list(statuses), list(entries), None

    def status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def search(self, **filters):
        self.filters = filters
        return iter(self.entries)


def entry(client="site-a", dataset="clinical-research", available=True):
    catalog = make_catalog(dataset=dataset) if available else None
    return CatalogEntry(
        client,
        dataset,
        available,
        "2026-06-01T00:00:00Z",
        catalog,
        1 if available else None,
        1 if available else None,
        "2026-06-01T00:00:00Z" if available else None,
    )


def parse(*argv):
    parser = argparse.ArgumentParser()
    catalog_cli._define_catalog_subcommands(parser)
    return parser.parse_args(list(argv))


class TestArguments:
    def test_show_options(self):
        args = parse("show", "--site", "a", "--dataset", "x")
        assert (args.study, args.site, args.dataset) == ("default", "a", "x")

    @pytest.mark.parametrize(
        "argv",
        [["show", "--modality", "m"], ["show", "--text", "t"], ["refresh"], ["read", "--wait"]],
    )
    def test_removed_and_unknown_options_are_rejected(self, argv):
        with pytest.raises(SystemExit):
            parse(*argv)


class TestRead:
    def test_read_options_have_defaults(self):
        args = parse("read")
        assert (args.study, args.sites, args.datasets, args.timeout, args.no_wait) == ("default", None, [], 300, False)

    def test_csv(self):
        assert catalog_cli._csv("a, b,a,") == ["a", "b"]
        assert catalog_cli._csv(None) is None
        for bad in ("", ",", " "):
            with pytest.raises(ValueError):
                catalog_cli._csv(bad)

    def test_dataset_options_become_a_per_site_mapping(self):
        assert catalog_cli._dataset_map(["a:x", "a:y", "b:z", "a:x"]) == {"a": ["x", "y"], "b": ["z"]}
        assert catalog_cli._dataset_map([]) is None
        for bad in ("x", ":x", "a:"):
            with pytest.raises(ValueError):
                catalog_cli._dataset_map([bad])

    def test_default_read_submits_the_exported_recipe_and_waits_without_downloading(self, monkeypatch):
        session = CatalogSession()
        seen = {}

        def connect(**kw):
            seen.update(kw)
            return session

        monkeypatch.setattr(cli_session, "new_cli_session_for_args", connect)
        args = parse("read", "--study", "s", "--sites", "a,b", "a:x")
        data = catalog_cli._submit_read(args)
        assert data == {"study": "s", "job_id": "job-1", "status": "FINISHED:COMPLETED"}
        assert seen["study"] == "s" and seen["args"] is args
        meta = session.submitted["meta"]
        assert meta["fdn_catalog_read"] is True
        assert meta["deploy_map"] == {"app_server": ["server"], "app_a": ["a"], "app_b": ["b"]}
        assert session.submitted["server"][0]["workflows"][0]["args"] == {"datasets": {"a": ["x"]}}
        assert session.closed and session.monitored
        assert not session.submitted["path"].exists()

    def test_no_wait_returns_submission_without_monitoring(self, monkeypatch):
        session = CatalogSession()

        def unexpected_monitor(*args, **kwargs):
            pytest.fail("--no-wait must not monitor the submitted job")

        session.monitor_job_and_return_job_meta = unexpected_monitor
        monkeypatch.setattr(cli_session, "new_cli_session_for_args", lambda **kw: session)
        data = catalog_cli._submit_read(parse("read", "--study", "s", "--no-wait"))

        assert data == {"study": "s", "job_id": "job-1"}
        assert session.closed and not session.submitted["path"].exists()

    @pytest.mark.parametrize("argv", [("--sites", ","), ("bad",), ("--timeout", "0")])
    def test_invalid_scope_is_rejected_before_connecting(self, monkeypatch, argv):
        def unexpected(**kw):
            pytest.fail("invalid arguments must not connect")

        monkeypatch.setattr(cli_session, "new_cli_session_for_args", unexpected)
        with pytest.raises(ValueError):
            catalog_cli._submit_read(parse("read", *argv))


@pytest.fixture
def machine_output():
    cli_output.set_output_format("json")
    try:
        yield
    finally:
        cli_output.set_output_format("txt")


@pytest.mark.parametrize("command", ["read", "show"])
@pytest.mark.parametrize(
    "error,code,error_code",
    [
        (AuthenticationError("bad login"), 2, "AUTH_FAILED"),
        (AuthorizationError("denied"), 1, "NOT_AUTHORIZED"),
        (NoConnection("offline"), 2, "CONNECTION_FAILED"),
        (SessionExpired("expired"), 2, "CONNECTION_FAILED"),
        (TimeoutError("late"), 3, "TIMEOUT"),
        (InvalidArgumentError("invalid"), 4, "INVALID_ARGS"),
        (InvalidJobDefinition("invalid job"), 4, "INVALID_ARGS"),
        (JobNotFound("missing"), 1, "JOB_NOT_FOUND"),
        (InternalError("broken"), 5, "INTERNAL_ERROR"),
        (RuntimeError("unexpected"), 5, "INTERNAL_ERROR"),
        (CommandError(error_code="DENIED", message="denied", exit_code=1), 1, "DENIED"),
    ],
)
def test_errors_use_shared_exit_codes_and_close_session(
    monkeypatch, capsys, machine_output, command, error, code, error_code
):
    session = CatalogSession(error=error)
    monkeypatch.setattr(cli_session, "new_cli_session_for_args", lambda **kw: session)
    with pytest.raises(SystemExit) as stopped:
        getattr(catalog_cli, "cmd_" + command)(parse(command))
    assert stopped.value.code == code and session.closed
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["schema_version"] == "1" and envelope["status"] == "error"
    assert envelope["exit_code"] == code and envelope["error_code"] == error_code


class TestShow:
    def args(self, **kw):
        return SimpleNamespace(**{"study": "s", "site": None, "dataset": None, **kw})

    def test_list_is_one_summary_row_per_dataset_with_site_health(self):
        client = FakeClient(statuses=[[health()]], entries=[entry(), entry("site-b", "imaging", available=False)])
        data = catalog_cli._show(client, self.args())
        assert client.filters == {} and data["total"] == 2
        assert (
            data["datasets"][0]["display_name"] == "Clinical research" and data["datasets"][0]["catalog_version"] == 1
        )
        assert data["datasets"][1] == {
            "site": "site-b",
            "dataset": "imaging",
            "available": False,
            "last_checked_at": "2026-06-01T00:00:00Z",
        }
        assert data["sites"] == [
            {
                "site": "site-a",
                "health": "OK",
                "reason_code": None,
                "last_success_at": "2026-06-01T00:00:20Z",
                "stale": False,
            }
        ]

    def test_filters_are_the_site_and_dataset_only(self):
        client = FakeClient(statuses=[[health()]])
        catalog_cli._show(client, self.args(site="site-a"))
        assert client.filters == {"client": "site-a"}

    def test_site_and_dataset_give_one_dataset_in_full(self):
        client = FakeClient(statuses=[[health()]], entries=[entry()])
        data = catalog_cli._show(client, self.args(site="site-a", dataset="clinical-research"))
        assert data["dataset"]["metadata"]["tables"][0]["name"] == "person"
        assert "datasets" not in data

    def test_unavailable_dataset_has_no_metadata(self):
        client = FakeClient(statuses=[[health()]], entries=[entry(available=False)])
        detail = catalog_cli._show(client, self.args(site="site-a", dataset="clinical-research"))["dataset"]
        assert "metadata" not in detail and detail["available"] is False

    def test_missing_dataset_is_an_error_with_a_hint(self):
        with pytest.raises(CommandError) as e:
            catalog_cli._show(FakeClient(statuses=[[health()]]), self.args(site="site-a", dataset="nope"))
        assert e.value.error_code == "DATASET_NOT_FOUND" and e.value.hint


class TestRendering:
    @pytest.fixture
    def out(self, monkeypatch):
        lines, ok = [], {}
        monkeypatch.setattr(catalog_cli, "print_human", lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
        monkeypatch.setattr(catalog_cli, "output_ok", lambda data, **k: ok.update(data=data, **k))
        return SimpleNamespace(lines=lines, ok=ok, monkeypatch=monkeypatch)

    def test_human_list_warns_about_failed_and_stale_sites(self, out):
        out.monkeypatch.setattr(catalog_cli, "is_json_mode", lambda: False)
        data = {
            "study": "s",
            "total": 1,
            "datasets": [
                {
                    "site": "a",
                    "dataset": "d",
                    "display_name": "D",
                    "available": True,
                    "catalog_version": 2,
                    "last_checked_at": "t",
                }
            ],
            "sites": [
                {"site": "a", "health": "FAILED", "reason_code": "TIMEOUT", "last_success_at": None, "stale": True},
                {"site": "b", "health": "OK", "reason_code": None, "last_success_at": "t", "stale": True},
                {"site": "c", "health": "OK", "reason_code": None, "last_success_at": "t", "stale": False},
            ],
        }
        catalog_cli._render_show(data)
        text = "\n".join(out.lines)
        assert "SITE" in text and "D" in text
        assert "warning: site a: last read failed (TIMEOUT)" in text and "site b: data may be out of date" in text
        assert "site c" not in text

    def test_json_mode_emits_the_data_unchanged(self, out):
        out.monkeypatch.setattr(catalog_cli, "is_json_mode", lambda: True)
        catalog_cli._render_show({"study": "s", "total": 0, "datasets": [], "sites": []})
        assert out.ok["data"]["total"] == 0 and out.lines == []

    def test_human_detail_shows_tables_and_columns(self, out):
        out.monkeypatch.setattr(catalog_cli, "is_json_mode", lambda: False)
        client = FakeClient(statuses=[[health()]], entries=[entry()])
        data = catalog_cli._show(client, SimpleNamespace(study="s", site="site-a", dataset="clinical-research"))
        catalog_cli._render_show(data)
        text = "\n".join(out.lines)
        assert "table person" in text and "person_id" in text and "int64" in text and "source: parquet" in text
