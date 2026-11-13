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

import io
import stat
import threading
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nvflare.private.fed.resource_stats import coordinator as coordinator_module
from nvflare.private.fed.resource_stats.archive_reader import WorkspaceResourceStatsError, WorkspaceResourceStatsReader
from nvflare.private.fed.resource_stats.coordinator import (
    RESOURCE_REPORT_ACCEPTED,
    RESOURCE_REPORT_CONFLICT,
    RESOURCE_REPORT_DUPLICATE,
    RESOURCE_REPORT_INVALID,
    RESOURCE_REPORT_NOT_EXPECTED,
    RESOURCE_REPORT_SERVER_ERROR,
    RESOURCE_REPORT_TOO_LATE,
    RESOURCE_SUMMARY_FILE,
    ResourceStatsCoordinator,
)
from nvflare.private.fed.resource_stats.measurements import canonical_json_bytes
from tests.resource_stats_report_test_utils import _participant_summary


def _participant(job_id="job-1", participant_name="site-1", reported_at="2026-09-17T12:00:00Z"):
    return _participant_summary(job_id, participant_name, reported_at)


def _workspace_zip(run_dir):
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        stats_dir = run_dir / "resource_stats"
        archive.write(stats_dir, stats_dir.relative_to(run_dir).as_posix())
        for path in sorted(stats_dir.rglob("*")):
            archive.write(path, path.relative_to(run_dir).as_posix())
    return stream.getvalue()


@pytest.fixture
def started_job(tmp_path):
    run_dir = tmp_path / "run_job-1"
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], run_dir)
    return coordinator, run_dir


@pytest.fixture
def published_job(started_job):
    coordinator, run_dir = started_job
    data = canonical_json_bytes(_participant())
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data}) == RESOURCE_REPORT_ACCEPTED
    )
    return run_dir, coordinator.finalize_job("job-1")


def test_accept_is_idempotent_and_finalized_bundle_reads_from_workspace(started_job):
    coordinator, run_dir = started_job
    data = canonical_json_bytes(_participant())

    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data}) == RESOURCE_REPORT_ACCEPTED
    )
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data})
        == RESOURCE_REPORT_DUPLICATE
    )
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": b"{}"}) == RESOURCE_REPORT_INVALID
    )
    conflict = canonical_json_bytes(_participant(reported_at="2026-09-17T12:00:01Z"))
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": conflict})
        == RESOURCE_REPORT_CONFLICT
    )
    summary_path = run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE
    assert not summary_path.exists()
    assert not (run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE).exists()

    # The accepted bytes live only in the parent ledger until finalization.
    # Job-owned file contents cannot replace the accepted in-memory measurements.
    summary_path.write_bytes(b"child-side mutation")

    coordinator.disable_clients("job-1", ["site-1"])
    summary = coordinator.finalize_job("job-1")
    assert summary["job_id"] == "job-1"
    assert [(entry["participant_name"], entry.get("status")) for entry in summary["participants"]] == [
        ("site-1", None),
        ("server", "missing"),
    ]
    assert summary["participants"][0]["reported_at"] == _participant()["reported_at"]
    assert {
        path.relative_to(run_dir).as_posix() for path in (run_dir / "resource_stats").rglob("*") if path.is_file()
    } == {
        "resource_stats/resource_summary.json",
    }
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data})
        == RESOURCE_REPORT_DUPLICATE
    )
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": conflict})
        == RESOURCE_REPORT_TOO_LATE
    )

    reader = WorkspaceResourceStatsReader(_workspace_zip(run_dir))
    assert reader.read_resource_summary() == summary
    assert reader.read_participant_summary("site-1") == _participant()
    coordinator.forget_job("job-1")
    assert not coordinator.has_job("job-1")


def test_summary_write_failure_occurs_at_finalization_not_acceptance(started_job):
    coordinator, run_dir = started_job
    data = canonical_json_bytes(_participant())

    with patch.object(ResourceStatsCoordinator, "_atomic_write_resource", side_effect=OSError("disk full")):
        assert (
            coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data})
            == RESOURCE_REPORT_ACCEPTED
        )
        with pytest.raises(OSError, match="disk full"):
            coordinator.finalize_job("job-1")

    assert not (run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE).exists()


def test_finalization_publishes_only_resource_summary(tmp_path):
    writes = []

    class RecordingCoordinator(ResourceStatsCoordinator):
        @classmethod
        def _atomic_write_resource(cls, run_dir, file_name, data):
            writes.append(file_name)
            return super()._atomic_write_resource(run_dir, file_name, data)

    run_dir = tmp_path / "run_job-1"
    coordinator = RecordingCoordinator()
    coordinator.start_job("job-1", ["site-1"], run_dir)
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())})

    coordinator.finalize_job("job-1")

    assert writes == [RESOURCE_SUMMARY_FILE]


def test_workspace_reader_accepts_archive_path_and_open_file(tmp_path, published_job):
    run_dir, summary = published_job
    archive_path = tmp_path / "workspace.zip"
    archive_path.write_bytes(_workspace_zip(run_dir))

    assert WorkspaceResourceStatsReader(archive_path).read_resource_summary() == summary
    with archive_path.open("rb") as archive_file:
        assert WorkspaceResourceStatsReader(archive_file).read_participant_summary("site-1") == _participant()


def test_start_job_rejects_resource_stats_symlink(tmp_path):
    run_dir = tmp_path / "run_job-1"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    (run_dir / "resource_stats").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        ResourceStatsCoordinator().start_job("job-1", ["site-1"], run_dir)

    assert list(outside.iterdir()) == []


def test_disable_clients_marks_participant_disabled_and_rejects_later_reports(tmp_path):
    run_dir = tmp_path / "run_job-1"
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1", "site-2"], run_dir)

    coordinator.disable_clients("job-1", ["site-2"])
    assert (
        coordinator.accept_resource_report("job-1", "site-2", {"participant_summary": b"anything"})
        == RESOURCE_REPORT_NOT_EXPECTED
    )

    summary = coordinator.finalize_job("job-1")
    statuses = {entry["participant_name"]: entry["status"] for entry in summary["participants"]}
    assert statuses["site-2"] == "disabled"
    assert statuses["site-1"] == "missing"


def test_restore_registration_purges_untrusted_pre_restart_reports(tmp_path):
    run_dir = tmp_path / "run_job-1"
    stale = run_dir / "resource_stats" / "participants" / "site-1.json"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(canonical_json_bytes(_participant()))

    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], run_dir, reset_existing=True)

    assert not stale.exists()
    summary = coordinator.finalize_job("job-1")
    site = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-1")
    assert site["status"] == "missing"


def test_restore_registration_replaces_existing_in_memory_ledger(started_job):
    coordinator, run_dir = started_job
    report = canonical_json_bytes(_participant())
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": report})
        == RESOURCE_REPORT_ACCEPTED
    )

    coordinator.start_job("job-1", ["site-1"], run_dir, reset_existing=True)
    summary = coordinator.finalize_job("job-1")

    site = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-1")
    assert site["status"] == "missing"


def test_duplicate_registration_rejects_different_expected_participants(started_job):
    coordinator, run_dir = started_job

    with pytest.raises(RuntimeError, match="registered differently"):
        coordinator.start_job("job-1", ["site-2"], run_dir)


def test_start_job_enforces_schema_identity_and_total_participant_bounds(tmp_path, monkeypatch):
    coordinator = ResourceStatsCoordinator()

    with pytest.raises(ValueError, match="job_id"):
        coordinator.start_job("../job", [], tmp_path / "bad-job")
    with pytest.raises(ValueError, match="client names"):
        coordinator.start_job("job-1", ["../site"], tmp_path / "bad-site")

    monkeypatch.setattr(coordinator_module, "MAX_PARTICIPANTS", 2)
    with pytest.raises(ValueError, match="cannot exceed 2"):
        coordinator.start_job("job-1", ["site-1", "site-2"], tmp_path / "too-many")


def test_finalization_preserves_receipt_time_without_stored_summary_times(tmp_path):
    coordinator = ResourceStatsCoordinator()
    run_dir = tmp_path / "run_job-1"
    coordinator.start_job("job-1", ["site-1"], run_dir)
    report = canonical_json_bytes(_participant())

    with patch.object(
        coordinator_module,
        "utc_timestamp",
        return_value="2026-09-17T12:00:01.000000Z",
    ):
        assert (
            coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": report})
            == RESOURCE_REPORT_ACCEPTED
        )
        summary = coordinator.finalize_job("job-1")

    accepted = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-1")
    assert accepted["received_at"] == "2026-09-17T12:00:01.000000Z"
    assert set(summary) == {"schema_version", "kind", "job_id", "participants"}


@pytest.mark.parametrize("removed", [False, True])
def test_repeated_finalization_returns_frozen_state_without_disk_io(tmp_path, removed):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1", "site-2"], tmp_path)
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())})
    first = coordinator.finalize_job("job-1")
    path = tmp_path / "resource_stats" / RESOURCE_SUMMARY_FILE
    if removed:
        path.unlink()
    else:
        path.write_bytes(b"changed after publication")
    coordinator.disable_clients("job-1", ["site-2"])
    assert (
        coordinator.accept_resource_report("job-1", "site-2", {"participant_summary": b"{}"})
        == RESOURCE_REPORT_TOO_LATE
    )
    with patch.object(coordinator, "_open_resource_directory", side_effect=AssertionError("unexpected disk I/O")):
        second = coordinator.finalize_job("job-1")
        assert second == first
        first["participants"].clear()
        assert coordinator.finalize_job("job-1") == second
    assert not path.exists() if removed else path.read_bytes() == b"changed after publication"


@pytest.mark.parametrize(
    "member,payloads",
    [
        ("resource_stats/README.txt", [b"{}"]),
        ("resource_stats/unexpected.json", [b"{}"]),
        ("resource_stats/staging/", [b""]),
        ("resource_stats/README.txt", [b"first", b"second"]),
    ],
)
def test_reader_ignores_unrequested_members(published_job, member, payloads):
    run_dir, summary = published_job
    workspace = io.BytesIO(_workspace_zip(run_dir))
    warning = pytest.warns(UserWarning, match="Duplicate name") if len(payloads) > 1 else nullcontext()
    with warning, ZipFile(workspace, "a", ZIP_DEFLATED) as archive:
        for payload in payloads:
            archive.writestr(member, payload)

    reader = WorkspaceResourceStatsReader(workspace.getvalue())
    assert reader.read_resource_summary() == summary
    assert reader.read_participant_summary("site-1") == _participant()


def test_reader_rejects_duplicate_requested_resource_summary(tmp_path):
    run_dir = tmp_path / "run_job-1"
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", [], run_dir)
    coordinator.finalize_job("job-1")

    workspace = io.BytesIO(_workspace_zip(run_dir))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(workspace, "a", ZIP_DEFLATED) as archive:
            archive.writestr("resource_stats/resource_summary.json", "{}")

    with pytest.raises(WorkspaceResourceStatsError, match="exactly one.*found 2"):
        WorkspaceResourceStatsReader(workspace.getvalue()).read_resource_summary()


def test_accept_rejects_report_that_would_exceed_per_job_payload_cap(tmp_path):
    run_dir = tmp_path / "run_job-1"
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1", "site-2"], run_dir)
    first = canonical_json_bytes(_participant(participant_name="site-1"))
    second = canonical_json_bytes(_participant(participant_name="site-2"))

    with patch.object(coordinator_module, "MAX_ACCEPTED_REPORT_BYTES_PER_JOB", len(first) + len(second) - 1):
        assert (
            coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": first})
            == RESOURCE_REPORT_ACCEPTED
        )
        assert (
            coordinator.accept_resource_report("job-1", "site-2", {"participant_summary": second})
            == RESOURCE_REPORT_SERVER_ERROR
        )

    summary = coordinator.finalize_job("job-1")
    statuses = {entry["participant_name"]: entry.get("status") for entry in summary["participants"]}
    assert statuses["site-1"] is None
    assert statuses["site-2"] == "missing"


def test_atomic_write_fsyncs_file_before_rename_and_directory_after(tmp_path, monkeypatch):
    target = tmp_path / "resource_stats" / "participant.json"
    events = []
    fsync = coordinator_module.os.fsync

    def record_fsync(descriptor):
        events.append((stat.S_ISDIR(coordinator_module.os.fstat(descriptor).st_mode), target.exists()))
        fsync(descriptor)

    monkeypatch.setattr(coordinator_module.os, "fsync", record_fsync)
    ResourceStatsCoordinator._atomic_write_resource(tmp_path, target.name, b"report")
    assert target.read_bytes() == b"report"
    assert events == [(False, False), (True, True)]


def test_finalizing_one_job_does_not_block_report_for_another_job(tmp_path):
    run_a = tmp_path / "run_job-a"
    run_b = tmp_path / "run_job-b"
    final_write_started = threading.Event()
    allow_final_write = threading.Event()

    class BlockingCoordinator(ResourceStatsCoordinator):
        @classmethod
        def _atomic_write_resource(cls, run_dir, file_name, data):
            if Path(run_dir) == run_a and file_name == RESOURCE_SUMMARY_FILE:
                final_write_started.set()
                if not allow_final_write.wait(timeout=5):
                    raise TimeoutError("test did not release the blocked finalization")
            return super()._atomic_write_resource(run_dir, file_name, data)

    coordinator = BlockingCoordinator()
    coordinator.start_job("job-a", [], run_a)
    coordinator.start_job("job-b", ["site-b"], run_b)
    report = canonical_json_bytes(_participant(job_id="job-b", participant_name="site-b"))
    finalize_errors = []
    report_results = []

    def finalize_a():
        try:
            coordinator.finalize_job("job-a")
        except Exception as exc:
            finalize_errors.append(exc)

    def report_b():
        report_results.append(coordinator.accept_resource_report("job-b", "site-b", {"participant_summary": report}))

    finalize_thread = threading.Thread(target=finalize_a)
    report_thread = threading.Thread(target=report_b)
    finalize_thread.start()
    report_finished_before_release = False
    try:
        assert final_write_started.wait(timeout=2)
        report_thread.start()
        report_thread.join(timeout=1)
        report_finished_before_release = not report_thread.is_alive()
    finally:
        allow_final_write.set()
        finalize_thread.join(timeout=5)
        if report_thread.ident is not None:
            report_thread.join(timeout=5)

    assert report_finished_before_release
    assert not finalize_thread.is_alive()
    assert not report_thread.is_alive()
    assert finalize_errors == []
    assert report_results == [RESOURCE_REPORT_ACCEPTED]


def test_finalization_cutoff_remains_atomic_for_reports_to_same_job(tmp_path):
    run_dir = tmp_path / "run_job-1"
    target_run_dir = run_dir
    final_write_started = threading.Event()
    allow_final_write = threading.Event()
    report_started = threading.Event()

    class BlockingCoordinator(ResourceStatsCoordinator):
        @classmethod
        def _atomic_write_resource(cls, run_dir, file_name, data):
            if Path(run_dir) == target_run_dir and file_name == RESOURCE_SUMMARY_FILE:
                final_write_started.set()
                if not allow_final_write.wait(timeout=5):
                    raise TimeoutError("test did not release the blocked finalization")
            return super()._atomic_write_resource(run_dir, file_name, data)

    coordinator = BlockingCoordinator()
    coordinator.start_job("job-1", ["site-1"], run_dir)
    report = canonical_json_bytes(_participant())
    finalize_results = []
    finalize_errors = []
    report_results = []

    def finalize():
        try:
            finalize_results.append(coordinator.finalize_job("job-1"))
        except Exception as exc:
            finalize_errors.append(exc)

    def report_after_cutoff():
        report_started.set()
        report_results.append(coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": report}))

    finalize_thread = threading.Thread(target=finalize)
    report_thread = threading.Thread(target=report_after_cutoff)
    finalize_thread.start()
    report_was_blocked = False
    try:
        assert final_write_started.wait(timeout=2)
        report_thread.start()
        assert report_started.wait(timeout=2)
        report_thread.join(timeout=0.2)
        report_was_blocked = report_thread.is_alive()
    finally:
        allow_final_write.set()
        finalize_thread.join(timeout=5)
        if report_thread.ident is not None:
            report_thread.join(timeout=5)

    assert report_was_blocked
    assert not finalize_thread.is_alive()
    assert not report_thread.is_alive()
    assert finalize_errors == []
    assert report_results == [RESOURCE_REPORT_TOO_LATE]
    site = next(entry for entry in finalize_results[0]["participants"] if entry["participant_name"] == "site-1")
    assert site["status"] == "missing"


def test_missing_filesystem_primitives_leave_no_registered_report(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator_module, "_SECURE_DIR_FD", False)
    coordinator = ResourceStatsCoordinator()
    with pytest.raises(OSError, match="directory-relative"):
        coordinator.start_job("job-1", ["site-1"], tmp_path)
    assert (
        coordinator.accept_resource_report("job-1", "site-1", canonical_json_bytes(_participant()))
        == RESOURCE_REPORT_NOT_EXPECTED
    )
    assert not (tmp_path / "resource_stats").exists()


def test_overflowing_total_preserves_accepted_reports_in_archive(tmp_path):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1", "site-2"], tmp_path)
    reports = [_participant_summary(participant_name=name) for name in ("site-1", "site-2")]
    for report in reports:
        report["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 1e308
        assert (
            coordinator.accept_resource_report(
                "job-1", report["participant_name"], {"participant_summary": canonical_json_bytes(report)}
            )
            == RESOURCE_REPORT_ACCEPTED
        )
    summary = coordinator.finalize_job("job-1")
    reader = WorkspaceResourceStatsReader(_workspace_zip(tmp_path))
    assert reader.read_resource_summary() == summary
    for report in reports:
        assert reader.read_participant_summary(report["participant_name"]) == report
