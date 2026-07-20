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
import os
import uuid

import pytest

from nvflare.tool.deploy.deploy_common import SLURM_STAGE_MANIFEST
from nvflare.tool.deploy.slurm_stage import SlurmStageError as ParentStagingError
from nvflare.tool.deploy.slurm_stage import stage_prepared_kit


def _make_prepared(tmp_path, marker="first", role="client", site="site-1"):
    prepared = tmp_path / f"prepared-{marker}"
    startup = prepared / "startup"
    local = prepared / "local"
    startup.mkdir(parents=True)
    local.mkdir()
    if role == "client":
        config_name = "fed_client.json"
        config = {"client": {"fqsn": site}}
    else:
        config_name = "fed_server.json"
        config = {"servers": [{"identity": site}]}
    (startup / config_name).write_text(json.dumps(config), encoding="utf-8")
    (local / "marker").write_text(marker, encoding="utf-8")
    sub_start = startup / "sub_start.sh"
    sub_start.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sub_start.chmod(0o700)
    return prepared


def _runtime_identity(workspace):
    return json.loads((workspace / ".nvflare_slurm" / "deployment.json").read_text())


def _identity_value(site="site-1"):
    return {"schema_version": 1, "deployment_uuid": str(uuid.uuid4()), "site": site}


def stage_runtime(prepared, workspace):
    if (prepared / "startup" / "fed_client.json").is_file():
        site = json.loads((prepared / "startup" / "fed_client.json").read_text())["client"]["fqsn"]
    else:
        site = json.loads((prepared / "startup" / "fed_server.json").read_text())["servers"][0]["identity"]
    (prepared / "local" / SLURM_STAGE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": site,
                "prepared_path": str(prepared.resolve()),
                "workspace_path": str(workspace.resolve(strict=False)),
            }
        ),
        encoding="utf-8",
    )
    stage_prepared_kit(prepared)
    return workspace / "kit" / "startup" / "sub_start.sh"


def test_stage_runtime_establishes_identity_kit_and_exact_links(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "shared" / "site-1"

    sub_start = stage_runtime(prepared, workspace)

    assert sub_start == workspace.resolve() / "kit" / "startup" / "sub_start.sh"
    identity = _runtime_identity(workspace)
    assert set(identity) == {"schema_version", "deployment_uuid", "site"}
    assert identity["schema_version"] == 1
    assert identity["site"] == "site-1"
    assert uuid.UUID(identity["deployment_uuid"]).version == 4
    assert (workspace / "kit" / "local" / "marker").read_text() == "first"
    assert os.readlink(workspace / "startup") == "kit/startup"
    assert os.readlink(workspace / "local") == "kit/local"


def test_stage_runtime_reprepared_kit_keeps_identity_and_replaces_kit(tmp_path):
    prepared = _make_prepared(tmp_path, marker="first")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)
    original_uuid = _runtime_identity(workspace)["deployment_uuid"]

    replacement = _make_prepared(tmp_path, marker="second")
    stage_runtime(replacement, workspace)
    assert _runtime_identity(workspace)["deployment_uuid"] == original_uuid
    assert (workspace / "kit" / "local" / "marker").read_text() == "second"
    assert not (workspace / "kit.previous").exists()


def test_stage_manifest_must_identify_canonical_prepared_output(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)
    manifest_path = prepared / "local" / SLURM_STAGE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_path"] = str(tmp_path / "other")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ParentStagingError, match="does not identify this canonical prepared output"):
        stage_prepared_kit(prepared)


def test_stage_reports_unresolvable_manifest_workspace_path(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)
    manifest_path = prepared / "local" / SLURM_STAGE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workspace_path"] = "/tmp/bad\x00path"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ParentStagingError, match="cannot resolve"):
        stage_prepared_kit(prepared)


def test_site_mismatch_does_not_modify_existing_kit(tmp_path):
    prepared = _make_prepared(tmp_path, marker="first")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)

    other = _make_prepared(tmp_path, marker="other", site="site-2")
    with pytest.raises(ParentStagingError, match="does not match the workspace identity"):
        stage_runtime(other, workspace)
    assert (workspace / "kit" / "local" / "marker").read_text() == "first"
    assert not (workspace / "kit.next").exists()


def test_stage_runtime_can_replace_client_with_server_kit_for_same_site(tmp_path):
    prepared = _make_prepared(tmp_path, marker="client")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)
    original_uuid = _runtime_identity(workspace)["deployment_uuid"]

    replacement = _make_prepared(tmp_path, marker="server", role="server")
    stage_runtime(replacement, workspace)

    assert _runtime_identity(workspace)["deployment_uuid"] == original_uuid
    assert (workspace / "kit" / "startup" / "fed_server.json").is_file()
    assert not (workspace / "kit" / "startup" / "fed_client.json").exists()
    assert (workspace / "kit" / "local" / "marker").read_text() == "server"


def test_stage_runtime_rejects_malformed_runtime_identity_before_kit_mutation(tmp_path):
    prepared = _make_prepared(tmp_path)
    identity = _identity_value()
    workspace = tmp_path / "workspace"
    control = workspace / ".nvflare_slurm"
    workspace.mkdir(mode=0o700)
    control.mkdir(mode=0o700)
    identity["deployment_uuid"] = 1
    (control / "deployment.json").write_text(json.dumps(identity), encoding="utf-8")
    existing_kit = workspace / "kit"
    existing_kit.mkdir()
    marker = existing_kit / "untouched"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(ParentStagingError, match="invalid workspace deployment identity"):
        stage_runtime(prepared, workspace)
    assert marker.read_text() == "original"
    assert not (workspace / "kit.next").exists()


def test_stage_runtime_rejects_unsafe_workspace_entries(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "safe-workspace"
    workspace.mkdir(mode=0o700)
    control = workspace / ".nvflare_slurm"
    control.mkdir(mode=0o700)
    (control / "deployment.json").write_text(json.dumps(_identity_value()), encoding="utf-8")
    (workspace / "startup").mkdir()
    with pytest.raises(ParentStagingError, match="relative symlink"):
        stage_runtime(prepared, workspace)
    assert not (workspace / "kit").exists()

    unidentified = tmp_path / "unidentified-workspace"
    unidentified.mkdir(mode=0o700)
    (unidentified / "old-run").mkdir()
    with pytest.raises(ParentStagingError, match="deployment data but has no identity"):
        stage_runtime(prepared, unidentified)
    assert not (unidentified / "kit").exists()


def test_stage_runtime_rejects_insecure_existing_workspace_and_control(tmp_path, monkeypatch):
    prepared = _make_prepared(tmp_path)

    workspace = tmp_path / "insecure-workspace"
    workspace.mkdir(mode=0o750)
    with pytest.raises(ParentStagingError, match="workspace must not grant group/world permissions"):
        stage_runtime(prepared, workspace)

    other_workspace = tmp_path / "other-owner-workspace"
    other_workspace.mkdir(mode=0o700)
    current_uid = os.geteuid()
    monkeypatch.setattr("nvflare.tool.deploy.slurm_stage.os.geteuid", lambda: current_uid + 1)
    with pytest.raises(ParentStagingError, match="workspace must be owned by current uid"):
        stage_runtime(prepared, other_workspace)
