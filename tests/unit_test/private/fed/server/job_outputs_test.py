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

import hashlib
import os
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.private.fed.server.job_outputs import filter_workspace_archive, outputs_only_refusal

WORKSPACE_FILES = {
    "app_server/FL_global_model.pt": b"model",
    "app_server/config/config_fed_server.json": b"{}",
    "metrics/metrics_summary.json": b"{}",
    "metrics/round_metrics.jsonl": b"",
    "log.txt": b"server log",
}


def _stored_archive(path, files=WORKSPACE_FILES, extra=None):
    with ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        for info, data in extra or []:
            z.writestr(info, data)
    return path


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _members(path):
    with ZipFile(path) as z:
        return sorted(z.namelist())


class TestFilterWorkspaceArchive:
    def test_keeps_only_declared_entries_at_their_paths(self, tmp_path):
        prepared = _stored_archive(tmp_path / "prepared.zip")
        destination = tmp_path / "workspace.zip"

        found = filter_workspace_archive(
            str(prepared), str(destination), ["app_server/FL_global_model.pt", "metrics/", "cross_site_val/"]
        )

        assert _members(destination) == [
            "app_server/FL_global_model.pt",
            "metrics/metrics_summary.json",
            "metrics/round_metrics.jsonl",
        ]
        with ZipFile(destination) as z:
            assert z.read("app_server/FL_global_model.pt") == b"model"
        assert found == ["app_server/FL_global_model.pt", "metrics/"]
        assert not prepared.exists()

    def test_never_writes_the_stored_archive_a_symlink_points_to(self, tmp_path):
        # FilesystemStorage prepares a download as a symlink to the stored archive
        store = tmp_path / "store"
        store.mkdir()
        stored = _stored_archive(store / "workspace")
        before = _digest(stored)
        prepared = tmp_path / "prepared.zip"
        os.symlink(stored, prepared)

        filter_workspace_archive(str(prepared), str(tmp_path / "workspace.zip"), ["metrics/"])

        assert stored.exists() and _digest(stored) == before
        assert len(_members(stored)) == len(WORKSPACE_FILES)  # a later full download still has every file
        assert not os.path.lexists(prepared)

    def test_rejects_symlink_and_escaping_members(self, tmp_path):
        link = ZipInfo("metrics/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        prepared = _stored_archive(
            tmp_path / "prepared.zip",
            files={"metrics/metrics_summary.json": b"{}"},
            extra=[(link, "/etc/passwd"), (ZipInfo("metrics/../../escape.txt"), b"x"), (ZipInfo("/abs"), b"x")],
        )
        destination = tmp_path / "workspace.zip"

        filter_workspace_archive(str(prepared), str(destination), ["metrics/", "escape.txt", "abs"])

        assert _members(destination) == ["metrics/metrics_summary.json"]

    def test_a_folder_entry_does_not_match_a_sibling_with_the_same_prefix(self, tmp_path):
        prepared = _stored_archive(tmp_path / "prepared.zip", files={"metrics2/a.json": b"", "metrics/b.json": b""})
        destination = tmp_path / "workspace.zip"

        filter_workspace_archive(str(prepared), str(destination), ["metrics/"])

        assert _members(destination) == ["metrics/b.json"]

    def test_failure_leaves_no_partial_file(self, tmp_path):
        prepared = tmp_path / "prepared.zip"
        prepared.write_bytes(b"not a zip")

        with pytest.raises(Exception):
            filter_workspace_archive(str(prepared), str(tmp_path / "workspace.zip"), ["metrics/"])

        assert sorted(p.name for p in tmp_path.iterdir()) == ["prepared.zip"]


class TestOutputsOnlyRefusal:
    def test_completed_job_with_outputs_is_filtered(self):
        meta = {
            JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value,
            JobMetaKey.JOB_OUTPUTS.value: ["metrics/", "metrics/"],
        }
        assert outputs_only_refusal(meta) == (None, ["metrics/"])

    @pytest.mark.parametrize(
        "meta, reason",
        [
            ({JobMetaKey.STATUS.value: RunStatus.FINISHED_EXECUTION_EXCEPTION.value}, "did not complete"),
            ({JobMetaKey.STATUS.value: RunStatus.FINISHED_ABORTED.value}, "did not complete"),
            ({JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value}, "declares no job_outputs"),
            (
                {JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value, JobMetaKey.JOB_OUTPUTS.value: ["../x"]},
                "invalid",
            ),
        ],
    )
    def test_other_jobs_get_the_whole_download(self, meta, reason):
        refusal, outputs = outputs_only_refusal(meta)
        assert reason in refusal and outputs == []
