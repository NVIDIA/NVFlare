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

import pytest

from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.private.fed.server.job_outputs import declared_outputs, outputs_only_refusal, stage_job_outputs


def _run_folder(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, files in os.walk(root) for f in files
    )


class TestStageJobOutputs:
    def test_moves_only_declared_files_out_of_the_workspace_at_their_paths(self, tmp_path):
        run_dir = _run_folder(
            tmp_path / "run",
            {
                "app_server/FL_global_model.pt": b"model",
                "app_server/config/config_fed_server.json": b"{}",
                "metrics/metrics_summary.json": b"{}",
                "log.txt": b"log",
            },
        )
        staging = tmp_path / "staging"

        staged = stage_job_outputs(
            [str(run_dir)], ["app_server/FL_global_model.pt", "metrics/", "plots/"], str(staging)
        )

        assert staged.found == ["app_server/FL_global_model.pt", "metrics/"]
        assert _files(staging) == ["app_server/FL_global_model.pt", "metrics/metrics_summary.json"]
        # moved, not copied: each file is stored once
        assert _files(run_dir) == ["app_server/config/config_fed_server.json", "log.txt"]

    def test_copies_log_files_the_log_commands_read(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"log.txt": b"log", "error_log.txt": b"err"})
        staging = tmp_path / "staging"

        stage_job_outputs([str(run_dir)], ["log.txt", "error_log.txt"], str(staging))

        assert _files(staging) == ["error_log.txt", "log.txt"]
        assert _files(run_dir) == ["error_log.txt", "log.txt"]

    def test_restore_puts_moved_files_back(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model", "log.txt": b"log"})
        staging = tmp_path / "staging"
        staged = stage_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt", "log.txt"], str(staging))

        staged.restore()

        assert _files(run_dir) == ["app_server/FL_global_model.pt", "log.txt"]
        assert (run_dir / "app_server" / "FL_global_model.pt").read_bytes() == b"model"

    def test_a_later_workspace_folder_replaces_an_earlier_ones_file(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"models/server.npy": b"run"})
        result_root = _run_folder(tmp_path / "result", {"models/server.npy": b"result"})
        staging = tmp_path / "staging"

        stage_job_outputs([str(run_dir), str(result_root)], ["models/server.npy"], str(staging))

        assert (staging / "models" / "server.npy").read_bytes() == b"result"
        assert (run_dir / "models" / "server.npy").read_bytes() == b"run"  # back in the workspace

    def test_stages_nothing_when_no_declared_output_exists(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"log.txt": b"log"})

        staged = stage_job_outputs([str(run_dir)], ["metrics/"], str(tmp_path / "staging"))

        assert staged.found == []
        assert _files(run_dir) == ["log.txt"]

    def test_skips_symlinks(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/metrics_summary.json": b"{}"})
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        os.symlink(secret, run_dir / "metrics" / "link.txt")
        staging = tmp_path / "staging"

        stage_job_outputs([str(run_dir)], ["metrics/"], str(staging))

        assert _files(staging) == ["metrics/metrics_summary.json"]

    def test_a_folder_entry_does_not_match_a_sibling_with_the_same_prefix(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics2/a.json": b"", "metrics/b.json": b""})
        staging = tmp_path / "staging"

        stage_job_outputs([str(run_dir)], ["metrics/"], str(staging))

        assert _files(staging) == ["metrics/b.json"]


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
