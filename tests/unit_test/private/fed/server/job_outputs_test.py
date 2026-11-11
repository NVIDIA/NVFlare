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
from nvflare.private.fed.server.job_outputs import declared_outputs, outputs_only_refusal, split_job_outputs


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


class TestSplitJobOutputs:
    def test_links_declared_files_and_the_rest_without_changing_the_workspace(self, tmp_path):
        run_dir = _run_folder(
            tmp_path / "run",
            {
                "app_server/FL_global_model.pt": b"model",
                "app_server/config/config_fed_server.json": b"{}",
                "metrics/metrics_summary.json": b"{}",
                "log.txt": b"log",
            },
        )
        before = _files(run_dir)

        split = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt", "metrics/", "plots/"])

        assert split.found == ["app_server/FL_global_model.pt", "metrics/"]
        assert _files(split.output_dirs[0]) == ["app_server/FL_global_model.pt", "metrics/metrics_summary.json"]
        assert _files(split.rest_dirs[0]) == ["app_server/config/config_fed_server.json", "log.txt"]
        assert _files(run_dir) == before  # nothing moved
        model = run_dir / "app_server" / "FL_global_model.pt"
        linked = os.path.join(split.output_dirs[0], "app_server", "FL_global_model.pt")
        assert os.stat(linked).st_ino == os.stat(model).st_ino  # the same file, not a copy

    def test_remove_deletes_only_links(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})
        split = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])

        split.remove()

        assert not any(os.path.exists(d) for d in split.output_dirs + split.rest_dirs)
        assert (run_dir / "app_server" / "FL_global_model.pt").read_bytes() == b"model"

    def test_log_files_the_log_commands_read_stay_in_the_workspace_too(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"log.txt": b"log", "error_log.txt": b"err", "plots/a.png": b""})

        split = split_job_outputs([str(run_dir)], ["log.txt", "error_log.txt", "plots/"])

        assert _files(split.output_dirs[0]) == ["error_log.txt", "log.txt", "plots/a.png"]
        assert _files(split.rest_dirs[0]) == ["error_log.txt", "log.txt"]

    def test_symlinks_stay_with_the_rest(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/metrics_summary.json": b"{}"})
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        os.symlink(secret, run_dir / "metrics" / "link.txt")

        split = split_job_outputs([str(run_dir)], ["metrics/"])

        assert _files(split.output_dirs[0]) == ["metrics/metrics_summary.json"]
        assert os.readlink(os.path.join(split.rest_dirs[0], "metrics", "link.txt")) == str(secret)

    def test_each_workspace_folder_is_split_beside_itself(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})
        result_root = _run_folder(tmp_path / "result", {"models/server.npy": b"npy", "notes.txt": b""})

        split = split_job_outputs([str(run_dir), str(result_root)], ["app_server/FL_global_model.pt", "models/"])

        assert [_files(d) for d in split.output_dirs] == [["app_server/FL_global_model.pt"], ["models/server.npy"]]
        assert [_files(d) for d in split.rest_dirs] == [[], ["notes.txt"]]
        assert all(os.path.dirname(d) == str(tmp_path) for d in split.output_dirs + split.rest_dirs)

    def test_returns_none_when_no_declared_output_exists(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"log.txt": b"log"})

        assert split_job_outputs([str(run_dir)], ["metrics/"]) is None
        assert sorted(os.listdir(tmp_path)) == ["run"]  # no link folders left

    def test_returns_none_and_leaves_no_links_when_linking_fails(self, tmp_path, monkeypatch):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})

        def cross_device(*args, **kwargs):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", cross_device)

        assert split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"]) is None
        assert sorted(os.listdir(tmp_path)) == ["run"]

    def test_links_left_by_an_interrupted_attempt_are_replaced(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})
        first = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])  # never removed

        second = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])

        assert first.output_dirs == second.output_dirs
        assert _files(second.output_dirs[0]) == ["app_server/FL_global_model.pt"]

    def test_a_folder_entry_does_not_match_a_sibling_with_the_same_prefix(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics2/a.json": b"", "metrics/b.json": b""})

        split = split_job_outputs([str(run_dir)], ["metrics/"])

        assert _files(split.output_dirs[0]) == ["metrics/b.json"]


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
