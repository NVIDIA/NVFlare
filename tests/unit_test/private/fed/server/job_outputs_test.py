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

import os
from zipfile import ZipFile

import pytest

from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.private.fed.server.job_outputs import archive_job_outputs, declared_outputs, outputs_only_refusal


def _run_folder(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _members(path):
    with ZipFile(path) as z:
        return sorted(z.namelist())


class TestArchiveJobOutputs:
    def test_zips_only_declared_files_at_their_workspace_paths(self, tmp_path):
        run_dir = _run_folder(
            tmp_path / "run",
            {
                "app_server/FL_global_model.pt": b"model",
                "app_server/config/config_fed_server.json": b"{}",
                "metrics/metrics_summary.json": b"{}",
                "metrics/round_metrics.jsonl": b"",
                "log.txt": b"log",
            },
        )
        destination = tmp_path / "outputs.zip"

        found = archive_job_outputs(
            [str(run_dir)], ["app_server/FL_global_model.pt", "metrics/", "cross_site_val/"], str(destination)
        )

        assert found == ["app_server/FL_global_model.pt", "metrics/"]
        assert _members(destination) == [
            "app_server/FL_global_model.pt",
            "metrics/metrics_summary.json",
            "metrics/round_metrics.jsonl",
        ]
        with ZipFile(destination) as z:
            assert z.read("app_server/FL_global_model.pt") == b"model"

    def test_merges_workspace_folders_at_the_archive_root_like_the_workspace_archive(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"run"})
        result_root = _run_folder(tmp_path / "result", {"models/server.npy": b"npy"})
        destination = tmp_path / "outputs.zip"

        found = archive_job_outputs(
            [str(run_dir), str(result_root)], ["app_server/FL_global_model.pt", "models/server.npy"], str(destination)
        )

        assert found == ["app_server/FL_global_model.pt", "models/server.npy"]
        assert _members(destination) == ["app_server/FL_global_model.pt", "models/server.npy"]

    def test_writes_nothing_when_no_declared_output_exists(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"log.txt": b"log"})
        destination = tmp_path / "outputs.zip"

        assert archive_job_outputs([str(run_dir)], ["metrics/"], str(destination)) == []
        assert not destination.exists()

    def test_skips_symlinks(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/metrics_summary.json": b"{}"})
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        os.symlink(secret, run_dir / "metrics" / "link.txt")
        destination = tmp_path / "outputs.zip"

        archive_job_outputs([str(run_dir)], ["metrics/"], str(destination))

        assert _members(destination) == ["metrics/metrics_summary.json"]

    def test_a_folder_entry_does_not_match_a_sibling_with_the_same_prefix(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics2/a.json": b"", "metrics/b.json": b""})
        destination = tmp_path / "outputs.zip"

        archive_job_outputs([str(run_dir)], ["metrics/"], str(destination))

        assert _members(destination) == ["metrics/b.json"]


class TestOutputsOnlyRefusal:
    def test_completed_job_with_outputs_is_served_its_outputs(self):
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


@pytest.mark.parametrize("value", [None, [], ["../x"], "metrics/"])
def test_declared_outputs_is_empty_for_missing_or_invalid_values(value):
    assert declared_outputs({JobMetaKey.JOB_OUTPUTS.value: value}) == []
