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

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pytest

import nvflare.app_opt.tracking.mlflow.mlflow_receiver as receiver_module
from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.fl_constant import ProcessType, ReservedKey
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.fuel.utils.config_service import ConfigService


def test_default_tracking_uri_uses_local_sqlite(tmp_path):
    result_root = tmp_path / "simulate_job"
    workspace = MagicMock()
    workspace.get_result_root.return_value = str(result_root)
    workspace.get_resources_file_path.return_value = None
    fl_ctx = MagicMock()
    fl_ctx.get_workspace.return_value = workspace
    fl_ctx.get_job_id.return_value = "simulate_job"

    receiver = MLflowReceiver()

    assert receiver._get_tracking_uri(fl_ctx) == f"sqlite:///{result_root / 'mlflow.db'}"
    workspace.get_result_root.assert_called_once_with("simulate_job")


@pytest.fixture
def tracking_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    result_root = tmp_path / "workspace" / "job"
    result_root.mkdir(parents=True)
    local = tmp_path / "workspace" / "local"
    local.mkdir()
    (tmp_path / "workspace" / "startup").mkdir()
    workspace = Workspace(str(tmp_path / "workspace"), "site-1")
    fl_ctx = MagicMock()
    fl_ctx.get_workspace.return_value = workspace
    fl_ctx.get_engine.return_value.get_workspace.return_value = workspace
    fl_ctx.get_job_id.return_value = "job"
    fl_ctx.get_prop.return_value = None
    fl_ctx.get_peer_context.return_value = None
    fl_ctx.get_process_type.return_value = ProcessType.CLIENT_JOB
    fl_ctx.get_identity_name.return_value = "site-1"
    return fl_ctx


def _set_job_study(fl_ctx, study, scope=None):
    meta = {"study": study, "scope": scope}
    fl_ctx.get_prop.side_effect = lambda key, default=None: meta if key == ReservedKey.JOB_META else default


def _set_study_uris(fl_ctx, uris):
    path = Path(fl_ctx.get_workspace().get_site_config_dir()) / "resources.json"
    path.write_text(json.dumps({"mlflow_study_tracking_uris": uris}))


@pytest.mark.parametrize("process_type", [ProcessType.SERVER_JOB, ProcessType.CLIENT_JOB])
@pytest.mark.parametrize("existing", [False, True])
def test_rejects_outside_sqlite_before_opening_backend(tracking_context, tmp_path, process_type, existing):
    outside = tmp_path / "outside.db"
    if existing:
        with sqlite3.connect(outside) as db:
            db.execute("CREATE TABLE marker (value TEXT)")
            db.execute("INSERT INTO marker VALUES ('unchanged')")
    before = outside.read_bytes() if existing else None
    tracking_context.get_process_type.return_value = process_type
    receiver = MLflowReceiver(tracking_uri=f"sqlite:///{outside}")

    with patch.object(receiver_module, "MlflowClient") as client:
        with pytest.raises(ValueError, match="tracking_uri"):
            receiver.initialize(tracking_context)
        client.assert_not_called()
    assert (outside.read_bytes() if outside.exists() else None) == before


@pytest.mark.parametrize("process_type", [ProcessType.SERVER_JOB, ProcessType.CLIENT_JOB])
@pytest.mark.parametrize("uri", ["http://127.0.0.1:5000", "https://tracking.example.invalid", "databricks"])
def test_unapproved_endpoint_cannot_receive_process_token(tracking_context, monkeypatch, process_type, uri):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "inert-process-token")
    tracking_context.get_process_type.return_value = process_type
    receiver = MLflowReceiver(tracking_uri=uri)

    with patch.object(receiver_module, "MlflowClient") as client, patch("requests.sessions.Session.send") as send:
        with pytest.raises(ValueError, match="tracking_uri"):
            receiver.initialize(tracking_context)
        client.assert_not_called()
        send.assert_not_called()


@pytest.mark.parametrize("scheme", ["sqlite", "file"])
def test_job_local_backend_is_allowed(tracking_context, scheme):
    result_root = Path(tracking_context.get_workspace().get_result_root("job"))
    path = result_root / ("metrics.db" if scheme == "sqlite" else "mlruns")
    uri = f"sqlite:///{path}" if scheme == "sqlite" else path.as_uri()

    assert MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context) == uri


@pytest.mark.parametrize(
    "uri",
    ["sqlite:///:memory:", "sqlite:///../outside.db", "sqlite:////tmp/test.db?uri=true", "file://other-host/mlruns"],
)
def test_unapproved_backend_forms_are_rejected(tracking_context, uri):
    with pytest.raises(ValueError, match="tracking_uri"):
        MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context)


@pytest.mark.parametrize("default_uri", [False, True])
def test_local_database_symlink_escape_is_rejected(tracking_context, tmp_path, default_uri):
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    (root / "mlflow.db").symlink_to(tmp_path / "outside.db")
    uri = None if default_uri else f"sqlite:///{root / 'mlflow.db'}"

    with pytest.raises(ValueError, match="must stay inside"):
        MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context)


def test_encoded_file_uri_escape_is_rejected(tracking_context):
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    with pytest.raises(ValueError, match="tracking_uri"):
        MLflowReceiver(tracking_uri=f"file://{root}/%2e%2e/outside")._get_tracking_uri(tracking_context)


@pytest.mark.parametrize("encoded_escape", ["%2e%2e/outside.db", "%2e%2e%2foutside.db"])
def test_encoded_sqlite_escape_is_rejected_before_driver_opens_the_path(tracking_context, encoded_escape):
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    receiver = MLflowReceiver(tracking_uri=f"sqlite:///{root}/{encoded_escape}")
    with pytest.raises(ValueError, match="must stay inside"):
        receiver_module.MlflowClient(tracking_uri=receiver._get_tracking_uri(tracking_context)).search_experiments()
    assert not (root.parent / "outside.db").exists()


def test_sqlite_driver_opens_the_canonical_path_with_percent_characters(tracking_context):
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    receiver = MLflowReceiver(tracking_uri=f"sqlite:///{root}/metrics%252e.db")
    receiver_module.MlflowClient(tracking_uri=receiver._get_tracking_uri(tracking_context)).search_experiments()
    assert (root / "metrics%2e.db").is_file()
    assert not (root / "metrics..db").exists()


@pytest.mark.parametrize("uris", ["*", ["*"], {"study-a": None}, {"study-a": "*"}])
def test_site_policy_does_not_allow_wildcards_or_malformed_entries(tracking_context, uris):
    _set_study_uris(tracking_context, uris)
    with pytest.raises(ValueError, match="mlflow_study_tracking_uris"):
        MLflowReceiver()._get_tracking_uri(tracking_context)


def test_job_config_cannot_approve_its_own_backend(tracking_context):
    uri = "https://tracking.example.invalid"
    _set_job_study(tracking_context, "study-a")
    app_config = Path(tracking_context.get_workspace().get_app_config_dir("job"))
    app_config.mkdir(parents=True)
    (app_config / "resources.json").write_text(json.dumps({"mlflow_study_tracking_uris": {"study-a": uri}}))
    # Even process-global application/resources sections are not the authority.
    with patch.object(ConfigService, "get_section", return_value={"mlflow_study_tracking_uris": {"study-a": uri}}):
        with pytest.raises(ValueError, match="tracking_uri"):
            MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context)


@pytest.mark.parametrize("artifact_location", ["../outside", "/tmp/outside"])
def test_artifact_location_is_confined_before_backend_creation(tracking_context, artifact_location):
    receiver = MLflowReceiver(artifact_location=artifact_location)
    with patch.object(receiver_module, "MlflowClient") as client:
        with pytest.raises(ValueError, match="artifact_location"):
            receiver.initialize(tracking_context)
        client.assert_not_called()


def test_artifact_directory_symlink_escape_is_rejected(tracking_context, tmp_path):
    root = Path(tracking_context.get_workspace().get_run_dir("job"))
    (root / "artifacts").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="artifact_location"):
        MLflowReceiver().initialize(tracking_context)


@pytest.mark.parametrize("process_type", [ProcessType.SERVER_JOB, ProcessType.CLIENT_JOB])
@pytest.mark.parametrize("study", ["study-a", "study-b"])
def test_study_routes_to_its_endpoint_without_a_job_uri(tracking_context, process_type, study):
    endpoints = {"study-a": "https://tracking-a.example", "study-b": "https://tracking-b.example"}
    _set_study_uris(tracking_context, endpoints)
    _set_job_study(tracking_context, study, scope="study-b" if study == "study-a" else "study-a")
    tracking_context.get_process_type.return_value = process_type
    tracking_context.get_engine().get_clients.return_value = [SimpleNamespace(name="site-1")]
    receiver = MLflowReceiver()
    with patch.object(receiver_module, "MlflowClient") as client, patch("mlflow.set_tracking_uri") as set_global:
        receiver.initialize(tracking_context)
        client.assert_called_once_with(tracking_uri=endpoints[study])
        set_global.assert_not_called()

    other_study = "study-b" if study == "study-a" else "study-a"
    with pytest.raises(ValueError, match="conflicts"):
        MLflowReceiver(tracking_uri=endpoints[other_study])._get_tracking_uri(tracking_context)
    assert MLflowReceiver(tracking_uri=endpoints[study])._get_tracking_uri(tracking_context) == endpoints[study]

    _set_job_study(tracking_context, "unconfigured")
    assert MLflowReceiver()._get_tracking_uri(tracking_context).endswith("/job/mlflow.db")


def test_default_sqlite_backend_records_metrics_without_http(tracking_context, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "inert-process-token")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://unapproved.example.invalid")
    receiver = MLflowReceiver()
    with patch("requests.sessions.Session.send", side_effect=AssertionError("unexpected HTTP request")):
        receiver.initialize(tracking_context)
        client = receiver.get_mlflow_client("site-1")
        run_id = receiver.get_run_id("site-1")
        client.log_metric(run_id, "accuracy", 0.75)
        data = AnalyticsData("note", "local text", AnalyticsDataType.TEXT, sender=LogWriterName.MLFLOW, path="note.txt")
        receiver.save(tracking_context, data.to_dxo().to_shareable(), "site-1")
        receiver.finalize(tracking_context)
        assert client.get_run(run_id).data.metrics["accuracy"] == 0.75
        artifact_root = Path(url2pathname(urlsplit(client.get_run(run_id).info.artifact_uri).path))
        assert (artifact_root / "note.txt").read_text() == "local text"
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    assert (root / "mlflow.db").read_bytes().startswith(b"SQLite format 3")


@pytest.mark.parametrize("destination", ["http", "outside_file"])
def test_job_supplied_experiment_cannot_redirect_artifacts(tracking_context, tmp_path, monkeypatch, destination):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "inert-process-token")
    root = Path(tracking_context.get_workspace().get_result_root("job"))
    backend_uri = f"sqlite:///{root / 'supplied.db'}"
    artifact_uri = "https://attacker.example.invalid/artifacts" if destination == "http" else tmp_path.as_uri()
    client = receiver_module.MlflowClient(tracking_uri=backend_uri)
    client.create_experiment("FLARE FL Experiment", artifact_location=artifact_uri)
    receiver = MLflowReceiver(tracking_uri=backend_uri)

    with patch("requests.sessions.Session.send") as send:
        with pytest.raises(ValueError, match="artifact"):
            receiver.initialize(tracking_context)
        send.assert_not_called()
    assert receiver.run_ids == {}


@pytest.mark.parametrize("destination", ["http", "outside_file", "outside_symlink", "symlink_parent"])
def test_text_upload_rechecks_run_artifact_destination(tracking_context, tmp_path, destination):
    receiver = MLflowReceiver()
    root = Path(tracking_context.get_workspace().get_run_dir("job"))
    artifacts = root / "artifacts"
    artifacts.mkdir()
    if destination == "symlink_parent":
        outside = tmp_path / "outside" / "cache"
        outside.mkdir(parents=True)
        (root / "link").symlink_to(outside, target_is_directory=True)
        artifact_uri = f"{root.as_uri()}/link/../leak"
    elif destination == "outside_symlink":
        (artifacts / "note.txt").symlink_to(tmp_path / "outside.txt")
        artifact_uri = artifacts.as_uri()
    else:
        artifact_uri = "https://attacker.example.invalid/artifacts" if destination == "http" else tmp_path.as_uri()
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(artifact_uri=artifact_uri))
    receiver.mlflow_clients["site-1"] = client
    receiver.run_ids["site-1"] = "run"
    data = AnalyticsData("note", "test text", AnalyticsDataType.TEXT, sender=LogWriterName.MLFLOW, path="note.txt")

    with pytest.raises(ValueError, match="artifact"):
        receiver.save(tracking_context, data.to_dxo().to_shareable(), "site-1")
    client.log_text.assert_not_called()


def test_site_resource_override_does_not_inherit_default_approvals(tracking_context):
    uri = "https://tracking.example.invalid"
    _set_job_study(tracking_context, "study-a")
    local = Path(tracking_context.get_workspace().get_site_config_dir())
    (local / "resources.json.default").write_text(json.dumps({"mlflow_study_tracking_uris": {"study-a": uri}}))
    assert MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context) == uri
    (local / "resources.json").write_text("{}")

    with pytest.raises(ValueError, match="tracking_uri"):
        MLflowReceiver(tracking_uri=uri)._get_tracking_uri(tracking_context)
