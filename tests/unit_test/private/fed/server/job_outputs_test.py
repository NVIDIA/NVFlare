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
from nvflare.private.fed.server.job_outputs import (
    declared_outputs,
    has_invalid_job_outputs,
    is_produced,
    link_job_outputs,
    outputs_only_refusal,
)


def _run_folder(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def split_job_outputs(source_dirs, outputs):
    split = link_job_outputs(source_dirs, outputs)
    if split:
        split.link_rest()
    return split


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

    @pytest.mark.parametrize("symlink_in", ["run", "result"])
    def test_an_output_path_also_kept_with_another_folders_rest_rejects_the_split(self, tmp_path, symlink_in):
        # the workspace folders share archive paths: the split must not change which copy a full download keeps
        run_dir = _run_folder(tmp_path / "run", {"app_server/x.json": b"{}"})
        result_root = _run_folder(tmp_path / "result", {})
        initial = tmp_path / "initial.npy"
        initial.write_bytes(b"initial")
        trained, linked = (result_root, run_dir) if symlink_in == "run" else (run_dir, result_root)
        _run_folder(trained, {"models/server.npy": b"trained"})
        (linked / "models").mkdir(parents=True)
        os.symlink(initial, linked / "models" / "server.npy")  # a symlink is never an output
        split = link_job_outputs([str(run_dir), str(result_root)], ["models/"])

        assert split.found == ["models/"]
        with pytest.raises(OSError):  # store the workspace whole
            split.link_rest()

    def test_the_same_output_path_in_two_folders_keeps_the_split(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"models/server.npy": b"initial"})
        result_root = _run_folder(tmp_path / "result", {"models/server.npy": b"trained"})

        split = split_job_outputs([str(run_dir), str(result_root)], ["models/"])

        # both in the outputs archive, in folder order, so the later folder's copy still wins
        assert [_files(d) for d in split.output_dirs] == [["models/server.npy"], ["models/server.npy"]]

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

    def test_a_relative_symlink_to_an_output_still_resolves_in_the_workspace(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/model.pt": b"model"})
        os.symlink("model.pt", run_dir / "app_server" / "latest.pt")

        split = split_job_outputs([str(run_dir)], ["app_server/model.pt"])

        latest = os.path.join(split.rest_dirs[0], "app_server", "latest.pt")
        with open(latest, "rb") as f:  # what the workspace archive reads, as before the split
            assert f.read() == b"model"

    def test_a_relative_symlink_through_a_linked_folder_reads_the_same_file(self, tmp_path):
        run_dir = _run_folder(
            tmp_path / "run",
            {
                "app_server/FL_global_model.pt": b"m",
                "models/model.bin": b"top",
                "models/snapshots/model.bin": b"snapshot",
                "models/snapshots/v1/weights.bin": b"",
            },
        )
        os.symlink("snapshots/v1", run_dir / "models" / "current")
        # "current/.." is "snapshots", not "models": normalizing the path would change the file it reads
        os.symlink("../models/current/../model.bin", run_dir / "app_server" / "latest.bin")
        with open(run_dir / "app_server" / "latest.bin", "rb") as f:
            original = f.read()

        split = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])

        with open(os.path.join(split.rest_dirs[0], "app_server", "latest.bin"), "rb") as f:
            assert f.read() == original == b"snapshot"

    def test_a_symlinked_folder_stays_a_link(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"data/a.csv": b"a", "app_server/FL_global_model.pt": b"m"})
        os.symlink("data", run_dir / "data_link")

        split = split_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])

        linked = os.path.join(split.rest_dirs[0], "data_link")
        assert os.path.islink(linked) and os.listdir(linked) == ["a.csv"]

    def test_a_source_replaced_right_after_its_link_was_made_rejects_the_split(self, tmp_path, monkeypatch):
        import nvflare.private.fed.server.job_outputs as job_outputs_module

        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"old"})
        model = run_dir / "app_server" / "FL_global_model.pt"
        real_link = job_outputs_module._link

        def link_then_replace(source, target):
            real_link(source, target)
            if source == str(model):
                model.unlink()
                model.write_bytes(b"newer")  # the stored link still names the old file

        monkeypatch.setattr(job_outputs_module, "_link", link_then_replace)
        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])
        monkeypatch.setattr(job_outputs_module, "_link", real_link)

        # the outputs archive would hold the old model and the workspace archive would leave out the newer one
        with pytest.raises(OSError):
            split.link_rest()

    def test_a_file_written_after_the_outputs_are_linked_is_kept_with_the_rest(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model", "log.txt": b"old"})
        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])
        (run_dir / "late.txt").write_text("late")  # e.g. written while the outputs were uploading
        (run_dir / "log.txt").unlink()
        (run_dir / "log.txt").write_text("new")  # replaced: a new file

        split.link_rest()

        assert _files(split.rest_dirs[0]) == ["late.txt", "log.txt"]
        with open(os.path.join(split.rest_dirs[0], "log.txt")) as f:
            assert f.read() == "new"

    def test_a_declared_file_written_after_the_outputs_are_linked_rejects_the_split(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/metrics_summary.json": b"{}"})
        split = link_job_outputs([str(run_dir)], ["metrics/"])
        (run_dir / "metrics" / "round_metrics.jsonl").write_text("late")  # flushed while the outputs were stored

        # the stored outputs archive lacks it, so an outputs-only download would miss it: store the workspace whole
        with pytest.raises(OSError):
            split.link_rest()

    def test_an_undeclared_symlink_matching_the_declaration_written_late_keeps_the_split(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/metrics_summary.json": b"{}"})
        split = link_job_outputs([str(run_dir)], ["metrics/"])
        os.symlink("metrics_summary.json", run_dir / "metrics" / "latest.json")  # a symlink is never an output

        split.link_rest()

        assert os.path.islink(os.path.join(split.rest_dirs[0], "metrics", "latest.json"))

    @pytest.mark.parametrize("change", ["replaced", "removed", "appended"])
    def test_a_stored_output_changed_afterwards_rejects_the_split(self, tmp_path, change):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"old"})
        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"])
        model = run_dir / "app_server" / "FL_global_model.pt"
        if change == "appended":
            with open(model, "ab") as f:  # the same file, more content (e.g. metrics written in append mode)
                f.write(b"more")
        else:
            model.unlink()
            if change == "replaced":
                model.write_bytes(b"new")  # a new file at the same path

        with pytest.raises(OSError):  # the stored outputs archive is out of date: store the workspace whole
            split.link_rest()

    @pytest.mark.parametrize("change", ["appended", "rotated"])
    def test_a_declared_log_written_meanwhile_rejects_the_split(self, tmp_path, change):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"m", "log.txt": b"old"})
        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt", "log.txt"])
        log = run_dir / "log.txt"
        if change == "appended":
            with open(log, "ab") as f:
                f.write(b" more")
        else:
            log.rename(run_dir / "log.txt.1")
            log.write_bytes(b"new")

        # the outputs archive holds the old copy, so an outputs-only download would be out of date
        with pytest.raises(OSError):
            split.link_rest()

    def test_a_user_file_named_like_a_log_in_a_declared_folder_is_checked_like_any_output(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics/log.json": b"{}"})
        split = link_job_outputs([str(run_dir)], ["metrics/"])
        (run_dir / "metrics" / "log.json").unlink()

        with pytest.raises(OSError):
            split.link_rest()

    def test_an_unchanged_declared_log_is_in_both_archives(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"m", "log.txt": b"log"})
        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt", "log.txt"])

        split.link_rest()

        assert "log.txt" in _files(split.output_dirs[0])
        assert _files(split.rest_dirs[0]) == ["log.txt"]  # where the log commands read it

    def test_an_unreadable_folder_falls_back_to_the_whole_workspace(self, tmp_path, monkeypatch):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})
        real_walk = os.walk

        def failing_walk(top, onerror=None, **kwargs):
            if onerror:
                onerror(PermissionError(13, "Permission denied", str(top)))
            return real_walk(top, **kwargs)

        monkeypatch.setattr(os, "walk", failing_walk)

        assert link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"]) is None
        assert sorted(os.listdir(tmp_path)) == ["run"]

    def test_a_folder_entry_does_not_match_a_sibling_with_the_same_prefix(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"metrics2/a.json": b"", "metrics/b.json": b""})

        split = split_job_outputs([str(run_dir)], ["metrics/"])

        assert _files(split.output_dirs[0]) == ["metrics/b.json"]


class TestOutputsOnlyRefusal:
    def test_a_completed_job_is_served_its_outputs(self):
        assert outputs_only_refusal({JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value}) is None

    @pytest.mark.parametrize("status", [RunStatus.FINISHED_EXECUTION_EXCEPTION, RunStatus.FINISHED_ABORTED])
    def test_a_job_that_did_not_complete_gets_the_whole_download(self, status):
        assert "did not complete" in outputs_only_refusal({JobMetaKey.STATUS.value: status.value})


@pytest.mark.parametrize("value, invalid", [(None, False), ([], False), (["metrics/"], False), (["../x"], True)])
def test_has_invalid_job_outputs(value, invalid):
    assert has_invalid_job_outputs({JobMetaKey.JOB_OUTPUTS.value: value}) is invalid


class TestProducedFiles:
    @pytest.mark.parametrize(
        "path",
        [
            "app_server/FL_global_model.pt",
            "models/server.npy",
            "metrics/round_metrics.jsonl",
            "cross_site_val/cross_val_results.json",
            "tb_events/site-1/events.out.tfevents.1",
            "app_server/custom/new_report.json",  # created by the job, not deployed with it
            "results/catalog.txt",
        ],
    )
    def test_files_the_job_wrote_are_produced(self, path):
        assert is_produced(path)

    @pytest.mark.parametrize(
        "path",
        [
            "meta.json",
            "fl_app.txt",
            "log.txt",
            "error_log.txt",
            "log.json",
            "log_fl.txt",
            "log.txt.3",  # rotated
            "task_1_log.txt",  # a task worker's prefixed log
            "audit.log",
            "stats_pool_summary.json",
            "server.stats_pool_records.csv",
            "app_server/custom/__pycache__/net.cpython-312.pyc",
            "job_cert/job.crt",
            "app_server/.__nvfl_sig.json",
        ],
    )
    def test_job_files_logs_and_system_files_are_not(self, path):
        assert not is_produced(path)

    def test_deployed_files_are_not(self):
        deployed = {"app_server/config/config_fed_server.json", "app_server/custom/net.py"}

        assert not is_produced("app_server/custom/net.py", deployed)
        assert is_produced("app_server/custom/other.py", deployed)


class TestLinkProducedOutputs:
    def test_without_job_outputs_every_produced_file_is_an_output(self, tmp_path):
        deployed = {"app_server/config/config_fed_server.json", "app_server/custom/net.py"}
        run_dir = _run_folder(
            tmp_path / "run",
            {
                "app_server/config/config_fed_server.json": b"{}",
                "app_server/custom/net.py": b"net",
                "app_server/custom/__pycache__/net.cpython-312.pyc": b"pyc",
                "app_server/FL_global_model.pt": b"model",
                "app_server/my_model.bin": b"custom persistor",
                "metrics/metrics_summary.json": b"{}",
                "meta.json": b"{}",
                "fl_app.txt": b"app",
                "log.txt": b"log",
            },
        )

        split = link_job_outputs([str(run_dir)], [], deployed)
        split.link_rest()

        produced = ["app_server/FL_global_model.pt", "app_server/my_model.bin", "metrics/metrics_summary.json"]
        assert split.found == produced
        assert _files(split.output_dirs[0]) == produced
        assert _files(split.rest_dirs[0]) == [
            "app_server/config/config_fed_server.json",
            "app_server/custom/__pycache__/net.cpython-312.pyc",
            "app_server/custom/net.py",
            "fl_app.txt",
            "log.txt",
            "meta.json",
        ]

    def test_job_outputs_narrow_the_outputs(self, tmp_path):
        run_dir = _run_folder(
            tmp_path / "run", {"app_server/FL_global_model.pt": b"model", "metrics/metrics_summary.json": b"{}"}
        )

        split = link_job_outputs([str(run_dir)], ["app_server/FL_global_model.pt"], set())

        assert _files(split.output_dirs[0]) == ["app_server/FL_global_model.pt"]

    def test_nothing_produced_keeps_the_workspace_whole(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/config/c.json": b"{}", "log.txt": b"log"})

        assert link_job_outputs([str(run_dir)], [], {"app_server/config/c.json"}) is None

    def test_a_file_produced_after_the_outputs_were_stored_rejects_the_split(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model"})
        split = link_job_outputs([str(run_dir)], [], set())
        (run_dir / "late_report.json").write_text("{}")  # an output missing from the stored archive

        with pytest.raises(OSError):
            split.link_rest()

    def test_a_server_log_written_after_the_outputs_were_stored_keeps_the_split(self, tmp_path):
        run_dir = _run_folder(tmp_path / "run", {"app_server/FL_global_model.pt": b"model", "log.txt": b"old"})
        split = link_job_outputs([str(run_dir)], [], set())
        with open(run_dir / "log.txt", "ab") as f:
            f.write(b" more")  # the server keeps logging; logs are not outputs

        split.link_rest()

        assert _files(split.rest_dirs[0]) == ["log.txt"]


@pytest.mark.parametrize("value", [None, [], ["../x"], "metrics/"])
def test_declared_outputs_is_empty_for_missing_or_invalid_values(value):
    assert declared_outputs({JobMetaKey.JOB_OUTPUTS.value: value}) == []
