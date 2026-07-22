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


def test_stage_runtime_installs_kit_and_exact_links(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "shared" / "site-1"

    sub_start = stage_runtime(prepared, workspace)

    assert sub_start == workspace.resolve() / "kit" / "startup" / "sub_start.sh"
    assert not (workspace / ".nvflare_slurm").exists()
    assert (workspace / "kit" / "local" / "marker").read_text() == "first"
    assert os.readlink(workspace / "startup") == "kit/startup"
    assert os.readlink(workspace / "local") == "kit/local"


def test_stage_runtime_reprepared_kit_replaces_kit(tmp_path):
    prepared = _make_prepared(tmp_path, marker="first")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)

    replacement = _make_prepared(tmp_path, marker="second")
    stage_runtime(replacement, workspace)
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


def test_stage_runtime_can_replace_kit_for_another_site(tmp_path):
    prepared = _make_prepared(tmp_path, marker="first")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)

    other = _make_prepared(tmp_path, marker="other", site="site-2")
    stage_runtime(other, workspace)

    assert (workspace / "kit" / "local" / "marker").read_text() == "other"
    assert not (workspace / "kit.next").exists()


def test_stage_runtime_can_replace_client_with_server_kit_for_same_site(tmp_path):
    prepared = _make_prepared(tmp_path, marker="client")
    workspace = tmp_path / "workspace"
    stage_runtime(prepared, workspace)

    replacement = _make_prepared(tmp_path, marker="server", role="server")
    stage_runtime(replacement, workspace)

    assert (workspace / "kit" / "startup" / "fed_server.json").is_file()
    assert not (workspace / "kit" / "startup" / "fed_client.json").exists()
    assert (workspace / "kit" / "local" / "marker").read_text() == "server"


def test_stage_runtime_rejects_unsafe_workspace_entries(tmp_path):
    prepared = _make_prepared(tmp_path)
    workspace = tmp_path / "safe-workspace"
    workspace.mkdir(mode=0o700)
    (workspace / "startup").mkdir()
    with pytest.raises(ParentStagingError, match="relative symlink"):
        stage_runtime(prepared, workspace)
    assert not (workspace / "kit").exists()


def test_stage_runtime_rejects_insecure_existing_workspace(tmp_path, monkeypatch):
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
