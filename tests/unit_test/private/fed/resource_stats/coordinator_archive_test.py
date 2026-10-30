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
import json
import threading
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from nvflare.private.fed.resource_stats import coordinator as coordinator_module
from nvflare.private.fed.resource_stats.archive_reader import WorkspaceResourceStatsError, WorkspaceResourceStatsReader
from nvflare.private.fed.resource_stats.contract import SCHEMA_VERSION, canonical_json_bytes
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


def _participant(job_id="job-1", participant_name="site-1", reported_at="2026-09-17T12:00:00Z"):
    return {
        "duration_seconds": "2",
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.participant_summary",
        "job_id": job_id,
        "participant_name": participant_name,
        "reported_at": reported_at,
        "resource_capacity": {"status": "unavailable", "issues": ["observation_incomplete"]},
        "resource_time": {"status": "unavailable", "issues": ["observation_incomplete"]},
        "workspace_filesystem": {"status": "unavailable", "issues": ["observation_incomplete"]},
        "resource_utilization": {
            "cpu_consumed": {"status": "unavailable", "issues": ["observation_incomplete"]},
            "retained_content": {"status": "unavailable", "issues": ["unsupported"]},
            "message_traffic": {"status": "unavailable", "issues": ["observation_incomplete"]},
        },
    }


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
    participant_path = run_dir / "resource_stats" / "participants" / "site-1.json"
    assert not participant_path.exists()
    assert not (run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE).exists()

    # The accepted bytes live only in the parent ledger until finalization.
    # Anything the job child places at the publication path is replaced by
    # those exact accepted bytes before the summary marker is written.
    participant_path.write_bytes(b"child-side mutation")

    summary = coordinator.finalize_job("job-1")
    assert summary["job_id"] == "job-1"
    assert [(entry["participant_name"], entry.get("status")) for entry in summary["participants"]] == [
        ("site-1", None),
        ("server", "missing"),
    ]
    assert participant_path.read_bytes() == data
    assert {
        path.relative_to(run_dir).as_posix() for path in (run_dir / "resource_stats").rglob("*") if path.is_file()
    } == {
        "resource_stats/participants/site-1.json",
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


def test_participant_write_failure_occurs_at_finalization_not_acceptance(started_job):
    coordinator, run_dir = started_job
    data = canonical_json_bytes(_participant())

    with patch.object(ResourceStatsCoordinator, "_atomic_write_resource", side_effect=OSError("disk full")):
        assert (
            coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": data})
            == RESOURCE_REPORT_ACCEPTED
        )
        with pytest.raises(OSError, match="disk full"):
            coordinator.finalize_job("job-1")

    assert not (run_dir / "resource_stats" / "participants" / "site-1.json").exists()
    assert not (run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE).exists()


def test_finalization_writes_resource_summary_last(tmp_path):
    writes = []

    class RecordingCoordinator(ResourceStatsCoordinator):
        @classmethod
        def _atomic_write_resource(cls, run_dir, file_name, data, include_participants=False):
            writes.append((file_name, include_participants))
            return super()._atomic_write_resource(run_dir, file_name, data, include_participants)

    run_dir = tmp_path / "run_job-1"
    coordinator = RecordingCoordinator()
    coordinator.start_job("job-1", ["site-1"], run_dir)
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())})

    coordinator.finalize_job("job-1")

    assert writes == [("site-1.json", True), (RESOURCE_SUMMARY_FILE, False)]


def test_workspace_reader_accepts_archive_path_and_open_file(tmp_path, published_job):
    run_dir, summary = published_job
    archive_path = tmp_path / "workspace.zip"
    archive_path.write_bytes(_workspace_zip(run_dir))

    assert WorkspaceResourceStatsReader(archive_path).read_resource_summary() == summary
    with archive_path.open("rb") as archive_file:
        assert WorkspaceResourceStatsReader(archive_file).read_participant_summary("site-1") == _participant()


@pytest.mark.parametrize("archive_version", ["0.0", "2.0", "3.0", "4.0"])
def test_unsupported_formats_are_rejected_by_live_and_archive_readers(started_job, archive_version):
    coordinator, run_dir = started_job
    legacy_participant = _participant()
    legacy_participant["schema_version"] = archive_version
    assert (
        coordinator.accept_resource_report(
            "job-1", "site-1", {"participant_summary": canonical_json_bytes(legacy_participant)}
        )
        == RESOURCE_REPORT_INVALID
    )
    assert (
        coordinator.accept_resource_report(
            "job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())}
        )
        == RESOURCE_REPORT_ACCEPTED
    )
    legacy_summary = coordinator.finalize_job("job-1")
    legacy_summary["schema_version"] = archive_version
    summary_path = run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE
    summary_path.write_bytes(canonical_json_bytes(legacy_summary))
    participant_path = run_dir / "resource_stats" / "participants" / "site-1.json"
    participant_path.write_bytes(canonical_json_bytes(legacy_participant))
    reader = WorkspaceResourceStatsReader(_workspace_zip(run_dir))
    with pytest.raises(WorkspaceResourceStatsError):
        reader.read_resource_summary()


@pytest.mark.parametrize(
    "participant_version",
    ["2.0", "4.0"],
)
def test_workspace_reader_checks_detail_version_only_when_detail_is_requested(started_job, participant_version):
    coordinator, run_dir = started_job
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())})
    coordinator.finalize_job("job-1")
    summary_path = run_dir / "resource_stats" / RESOURCE_SUMMARY_FILE
    participant_path = run_dir / "resource_stats" / "participants" / "site-1.json"
    summary = json.loads(summary_path.read_bytes())
    participant = json.loads(participant_path.read_bytes())
    participant["schema_version"] = participant_version
    summary_path.write_bytes(canonical_json_bytes(summary))
    participant_path.write_bytes(canonical_json_bytes(participant))

    reader = WorkspaceResourceStatsReader(_workspace_zip(run_dir))
    # Job/study views trust the validated, self-contained rollup. Participant
    # body consistency is checked lazily for a site-detail request.
    reader.read_resource_summary()
    with pytest.raises(WorkspaceResourceStatsError):
        reader.read_participant_summary("site-1")


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


def test_disable_clients_is_a_no_op_for_unknown_or_already_accepted_participants(started_job):
    coordinator, run_dir = started_job
    report = canonical_json_bytes(_participant())
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": report})
        == RESOURCE_REPORT_ACCEPTED
    )

    # An already-accepted participant must not be retroactively disabled, and an
    # unknown job_id / participant name must not raise.
    coordinator.disable_clients("job-1", ["site-1"])
    coordinator.disable_clients("no-such-job", ["site-9"])

    summary = coordinator.finalize_job("job-1")
    site = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-1")
    assert "status" not in site


def test_forget_job_removes_registration_and_is_a_no_op_for_unknown_job(started_job):
    coordinator, run_dir = started_job
    assert coordinator.has_job("job-1")

    coordinator.forget_job("job-1")

    assert not coordinator.has_job("job-1")
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": b"anything"})
        == RESOURCE_REPORT_NOT_EXPECTED
    )
    # Forgetting a job that was never (or is no longer) registered must not raise.
    coordinator.forget_job("job-1")
    coordinator.forget_job("no-such-job")


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


def test_finalize_job_is_idempotent_and_rereads_the_published_summary(started_job):
    coordinator, run_dir = started_job
    report = canonical_json_bytes(_participant())
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": report})

    first = coordinator.finalize_job("job-1")
    with patch.object(coordinator_module.ResourceStatsCoordinator, "_atomic_write_resource") as atomic_write_resource:
        second = coordinator.finalize_job("job-1")

    # The second call must take the already-finalized short-circuit: it re-reads
    # the published summary from disk rather than re-running the write path.
    atomic_write_resource.assert_not_called()
    assert second == first


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


@pytest.mark.parametrize("copies", [0, 2])
def test_reader_checks_participant_member_count_only_when_requested(published_job, copies):
    run_dir, summary = published_job
    if copies == 0:
        (run_dir / "resource_stats" / "participants" / "site-1.json").unlink()
    workspace = io.BytesIO(_workspace_zip(run_dir))
    if copies == 2:
        with pytest.warns(UserWarning, match="Duplicate name"), ZipFile(workspace, "a", ZIP_DEFLATED) as archive:
            archive.writestr("resource_stats/participants/site-1.json", "{}")

    reader = WorkspaceResourceStatsReader(workspace.getvalue())
    assert reader.read_resource_summary() == summary
    with pytest.raises(WorkspaceResourceStatsError, match=f"exactly one.*found {copies}"):
        reader.read_participant_summary("site-1")


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


@pytest.mark.parametrize("field", ["cpu_consumed", "message_traffic"])
def test_reader_rejects_participant_values_that_disagree_with_resource_summary(started_job, field):
    coordinator, run_dir = started_job
    coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(_participant())})
    coordinator.finalize_job("job-1")

    participant_path = run_dir / "resource_stats" / "participants" / "site-1.json"
    participant = _participant()
    participant["resource_utilization"][field] = (
        {"sent_bytes": "1", "received_bytes": "2"}
        if field == "message_traffic"
        else {"status": "unavailable", "issues": ["attribution_incomplete"]}
    )
    participant_data = canonical_json_bytes(participant)
    participant_path.write_bytes(participant_data)

    reader = WorkspaceResourceStatsReader(_workspace_zip(run_dir))
    reader.read_resource_summary()
    with pytest.raises(WorkspaceResourceStatsError, match="values do not match"):
        reader.read_participant_summary("site-1")


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


def test_atomic_write_fsyncs_parent_directory_after_replace(tmp_path):
    target = tmp_path / "resource_stats" / "participant.json"

    with patch.object(ResourceStatsCoordinator, "_fsync_directory") as fsync_directory:
        ResourceStatsCoordinator._atomic_write(target, b"report")

    assert target.read_bytes() == b"report"
    fsync_directory.assert_called_once_with(Path(target.parent))


def test_finalizing_one_job_does_not_block_report_for_another_job(tmp_path):
    run_a = tmp_path / "run_job-a"
    run_b = tmp_path / "run_job-b"
    final_write_started = threading.Event()
    allow_final_write = threading.Event()

    class BlockingCoordinator(ResourceStatsCoordinator):
        @classmethod
        def _atomic_write_resource(cls, run_dir, file_name, data, include_participants=False):
            if Path(run_dir) == run_a and file_name == RESOURCE_SUMMARY_FILE:
                final_write_started.set()
                if not allow_final_write.wait(timeout=5):
                    raise TimeoutError("test did not release the blocked finalization")
            return super()._atomic_write_resource(run_dir, file_name, data, include_participants)

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
        def _atomic_write_resource(cls, run_dir, file_name, data, include_participants=False):
            if Path(run_dir) == target_run_dir and file_name == RESOURCE_SUMMARY_FILE:
                final_write_started.set()
                if not allow_final_write.wait(timeout=5):
                    raise TimeoutError("test did not release the blocked finalization")
            return super()._atomic_write_resource(run_dir, file_name, data, include_participants)

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
