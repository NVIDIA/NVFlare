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

import contextlib
import io
import json
import os
import queue
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import StorageException
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    ManifestEntry,
    ResultStatus,
    Selection,
    build_manifest,
    build_result,
    job_key,
    manifest_mac,
)
from nvflare.app_common.fdn.catalog.server import manager as manager_module
from nvflare.app_common.fdn.catalog.server.artifacts import CatalogArtifacts
from nvflare.app_common.fdn.catalog.server.commands import NotReady
from nvflare.app_common.fdn.catalog.server.manager import CatalogManager
from nvflare.app_common.fdn.catalog.server.reconcile import Reconciler
from nvflare.app_common.fdn.catalog.server.search import SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore
from nvflare.app_common.storages.filesystem_storage import _write_multi
from nvflare.private.fed.server.job_runner import JobRunner
from tests.timing_utils import ManualClock, ObservedLock, ThreadGroup
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog

STUDY = "study1"
KEY = b"k" * 32
T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


class FakeJobManager:
    def __init__(self):
        self.metas, self.manifests, self.raw = {}, {}, {}
        self.workspaces = {}

    def add(self, job_id, manifest=None, study=STUDY, marker=True, deploy_map=None, status="RUNNING"):
        meta = {"submit_time": 100.0, "status": status, "deploy_map": deploy_map or {"app": ["@ALL"]}}
        if study is not None:
            meta["study"] = study
        if marker:
            meta[CATALOG_JOB_META] = True
        self.metas[job_id] = meta
        if manifest is not None:
            self.manifests[job_id] = manifest

    def get_job(self, job_id, fl_ctx):
        return SimpleNamespace(job_id=job_id, meta=self.metas[job_id]) if job_id in self.metas else None

    def get_jobs_by_status(self, statuses, fl_ctx):
        return [SimpleNamespace(job_id=j, meta=m) for j, m in self.metas.items() if m["status"] in statuses]

    def get_storage_component(self, job_id, component, fl_ctx):
        assert component == "workspace"  # the name JobDefManager.save_workspace stores the archive under
        if job_id in self.workspaces:
            return self.workspaces[job_id]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            if job_id in self.manifests:
                m = self.manifests[job_id]
                zf.writestr(
                    "catalog_results.json",
                    json.dumps({"manifest": m, "mac": manifest_mac(job_key(KEY, job_id), job_id, m)}),
                )
            for name, content in self.raw.get(job_id, {}).items():
                zf.writestr(name, content)
        return buffer.getvalue()

    def save_workspace(self, job_id, data, fl_ctx):
        self.workspaces[job_id] = data

    def list_components(self, job_id, fl_ctx):
        archived = job_id in self.workspaces or job_id in self.manifests or job_id in self.raw
        return ["data", "meta", "workspace"] if archived else ["data", "meta"]


@pytest.fixture
def mgr(tmp_path):
    jobs = FakeJobManager()
    manager = CatalogManager(stale_after_s=3600, register_commands=False)
    manager._engine = SimpleNamespace(get_component=lambda name: jobs, new_context=lambda: contextlib.nullcontext())
    manager._store = CatalogStore(str(tmp_path))
    manager._artifacts = CatalogArtifacts(str(tmp_path))
    manager._workspace = SimpleNamespace(get_root_dir=lambda: str(tmp_path))
    manager._reconciler = Reconciler(manager._store, LIMITS)
    manager._now = lambda: T0
    manager._key = KEY
    manager.jobs = jobs
    yield manager
    manager._stop.set()
    if manager._thread is not None:
        manager._thread.join(timeout=3.0)
        assert not manager._thread.is_alive()


def held(mgr, job_id, manifest=None, study=STUDY, **kw):
    """A catalog job that was launched with its Study's context, so its result is the one to apply."""
    mgr.jobs.add(job_id, manifest, study=study, **kw)
    mgr._in_flight[study or "default"] = job_id


def ok_entry(**kw):
    return ManifestEntry("OK", build_result(ResultStatus.OK, [make_catalog(**kw)]))


def manifest(**entries):
    return build_manifest(entries)


def state(mgr, client, study=STUDY):
    return mgr._store.client_state(study, client)


def seed(mgr, study=STUDY, client="site-a"):
    mgr._reconciler.apply(study, client, Selection(None), build_result(ResultStatus.OK, [make_catalog()]), "t")


def event(mgr, event_type, **props):
    ctx = FLContext()
    for key, value in props.items():
        ctx.set_prop(key, value, private=True, sticky=False)
    mgr.handle_event(event_type, ctx)
    return ctx


def catalog_meta(job_id, study=STUDY):
    return {"job_id": job_id, "study": study, CATALOG_JOB_META: True}


class TestOneReadPerStudy:
    """One catalog job per Study runs at a time, so results apply in order and no read races another."""

    @pytest.fixture
    def run_dirs(self, mgr, tmp_path):
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(tmp_path / "run" / job_id))
        return lambda job_id: (tmp_path / "run" / job_id).mkdir(parents=True) or tmp_path / "run" / job_id

    def test_a_submit_is_refused_while_the_studys_read_is_running(self, mgr):
        mgr._in_flight[STUDY] = "j1"
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert "job j1" in ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    def test_a_submit_is_refused_while_an_earlier_one_waits_to_launch(self, mgr):
        mgr.jobs.add("j1", status="SUBMITTED")
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert "job j1" in ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    @pytest.mark.parametrize(
        "meta", [catalog_meta(None, study="other"), {"study": STUDY}]  # another Study; not a catalog job
    )
    def test_other_studies_and_other_jobs_are_not_refused(self, mgr, meta):
        mgr._in_flight[STUDY] = "j1"
        mgr.jobs.add("j1")
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: meta})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_a_finished_read_does_not_refuse_the_next(self, mgr):
        mgr.jobs.add("j1", status="FINISHED:COMPLETED")
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_only_the_first_job_launched_for_a_study_gets_a_context(self, mgr, run_dirs):
        first, second, elsewhere = run_dirs("j1"), run_dirs("j2"), run_dirs("j3")
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j1")})
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j2")})
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j3", study="other")})
        assert (first / "catalog_context.json").exists() and not (second / "catalog_context.json").exists()
        assert (elsewhere / "catalog_context.json").exists() and mgr._in_flight == {STUDY: "j1", "other": "j3"}

    def test_a_job_launched_while_another_holds_the_study_is_blocked_not_run_without_a_key(self, mgr, run_dirs):
        run_dirs("j1"), run_dirs("j2")
        first = event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j1")})
        second = event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j2")})
        assert first.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert "job j1" in second.get_prop(FLContextKey.JOB_BLOCK_REASON)
        assert mgr._in_flight == {STUDY: "j1"}

    def test_the_next_read_may_launch_once_the_result_is_applied(self, mgr, run_dirs):
        run_dirs("j1"), run_dirs("j2")
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry()}))
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j1")})
        mgr._ingest("j1")
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("j2")})
        assert mgr._in_flight == {STUDY: "j2"}

    def test_the_result_of_a_job_refused_at_launch_is_never_applied(self, mgr):
        mgr._in_flight[STUDY] = "j1"
        mgr.jobs.add("j2", manifest(**{"site-a": ok_entry()}))  # even if it carried a valid manifest
        mgr._ingest("j2")
        assert not mgr._store._clients and mgr._in_flight == {STUDY: "j1"}

    def test_a_job_that_survived_a_restart_still_holds_its_studys_read(self, mgr, tmp_path):
        mgr.jobs.add("j1", status="RUNNING")
        mgr.jobs.add("j0", study="other", status="SUBMITTED")
        mgr.jobs.add("done", status="FINISHED:COMPLETED")
        ctx = SimpleNamespace(
            get_engine=lambda: mgr._engine, get_workspace=lambda: SimpleNamespace(get_root_dir=lambda: str(tmp_path))
        )
        mgr._in_flight.clear()
        mgr._start(ctx)
        mgr._stop.set()
        assert mgr._in_flight == {STUDY: "j1"}

    def test_a_pending_job_aborted_after_restart_does_not_hold_the_study(self, mgr, tmp_path):
        mgr.jobs.add("pending", status="SUBMITTED")
        ctx = SimpleNamespace(get_engine=lambda: mgr._engine, get_workspace=lambda: mgr._workspace)
        mgr._start(ctx)
        # abort_before_start only sets terminal status; there is no lifecycle event to release a holder.
        mgr.jobs.metas["pending"]["status"] = "FINISHED:ABORTED"
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None and not mgr._in_flight

    @pytest.mark.parametrize("status", ["FINISHED:ABORTED", "FINISHED:FAILED_TO_RUN"])
    @pytest.mark.parametrize("gone", [False, True])
    def test_a_terminal_or_deleted_holder_is_recovered_without_an_event(self, mgr, gone, status):
        held(mgr, "j1", status=status)
        if gone:
            del mgr.jobs.metas["j1"]
        mgr._recover_finished()
        assert not mgr._in_flight


class TestHandOver:
    """The server puts the job's key and the names it holds in the job's run dir: a job process in a container
    sees its own job directory and nothing of the server's catalog store."""

    def launch(self, mgr, tmp_path, meta):
        run_dir = str(tmp_path / "run" / "j1")
        os.makedirs(run_dir, exist_ok=True)
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: run_dir)
        event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: meta})
        return os.path.join(run_dir, "catalog_context.json")

    def test_a_catalog_job_gets_its_own_key_and_the_names_of_its_study(self, mgr, tmp_path):
        seed(mgr)
        path = self.launch(mgr, tmp_path, catalog_meta("j1"))
        context = json.loads(open(path).read())
        assert bytes.fromhex(context["key"]) == job_key(KEY, "j1") and context["key"] != KEY.hex()
        assert context["previous"] == {"site-a": ["clinical-research"]}
        assert os.stat(path).st_mode & 0o077 == 0  # the key is readable by the server's user only

    def test_a_job_that_is_not_a_catalog_job_gets_nothing(self, mgr, tmp_path):
        assert not os.path.exists(self.launch(mgr, tmp_path, {"job_id": "j1", "study": STUDY}))

    def test_the_names_of_another_study_are_not_handed_over(self, mgr, tmp_path):
        seed(mgr, study="other", client="site-z")
        path = self.launch(mgr, tmp_path, catalog_meta("j1"))
        assert json.loads(open(path).read())["previous"] == {}


class TestTakeBack:
    """Before a run dir is archived for download, the manager removes the context it handed over."""

    def archive(self, mgr, tmp_path, *files, marker=True):
        mgr.jobs.add("j1", marker=marker)
        run_dir = tmp_path / "run" / "j1"
        run_dir.mkdir(parents=True)
        for name in files:
            (run_dir / name).write_text("{}")
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(run_dir))
        ctx = event(mgr, EventType.BEFORE_JOB_WORKSPACE_ARCHIVE, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        return run_dir, ctx

    def test_the_context_and_a_partial_write_of_it_are_removed_and_nothing_else(self, mgr, tmp_path):
        run_dir, ctx = self.archive(mgr, tmp_path, "catalog_context.json", "catalog_context.json.tmp", "keep.json")
        assert sorted(p.name for p in run_dir.iterdir()) == ["keep.json"]
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_another_jobs_file_of_the_same_name_is_left_alone(self, mgr, tmp_path):
        run_dir, ctx = self.archive(mgr, tmp_path, "catalog_context.json", marker=False)
        assert (run_dir / "catalog_context.json").exists() and ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_a_context_that_cannot_be_removed_blocks_the_archive(self, mgr, tmp_path, monkeypatch):
        def denied(path, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "unlink", denied)
        _, ctx = self.archive(mgr, tmp_path, "catalog_context.json")
        assert "could not be secured" in ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    def test_result_is_captured_privately_and_never_enters_the_downloadable_archive(self, mgr, tmp_path, monkeypatch):
        publication = manifest(**{"site-a": ok_entry(owner="private-after-withdrawal")})
        held(mgr, "j1")
        run_dir = tmp_path / "j1"
        run_dir.mkdir()
        result_path = run_dir / "catalog_results.json"
        result_path.write_text(
            json.dumps({"manifest": publication, "mac": manifest_mac(job_key(KEY, "j1"), "j1", publication)})
        )
        (run_dir / "catalog_context.json").write_text("private context")
        (run_dir / "log.txt").write_text("safe log")
        mgr._workspace = SimpleNamespace(
            get_run_dir=lambda job_id: str(run_dir),
            get_result_root=lambda job_id: str(run_dir),
            get_log_root=lambda job_id: str(run_dir),
            get_audit_root=lambda job_id: str(run_dir),
        )
        mgr._engine.get_workspace = lambda: mgr._workspace
        mgr._engine.fire_event = mgr.handle_event
        archive = tmp_path / "workspace.zip"

        def save_workspace(job_id, sources, fl_ctx):
            _write_multi(str(archive), sources)
            mgr.jobs.workspaces[job_id] = archive.read_bytes()
            return str(archive)

        monkeypatch.setattr(mgr.jobs, "save_workspace", save_workspace)
        ctx = FLContext()
        ctx.set_prop(ReservedKey.ENGINE, mgr._engine, private=True, sticky=False)
        ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "j1", private=True, sticky=False)
        # Exercise the actual runner's capture/archive/source-cleanup ordering with the real serializer.
        JobRunner(workspace_root=str(tmp_path))._save_workspace(ctx)
        assert not run_dir.exists()
        with zipfile.ZipFile(archive) as zf:
            assert zf.namelist() == ["log.txt"]
        private_result = tmp_path / "fdn" / "catalog-results" / "j1.json"
        assert private_result.stat().st_mode & 0o077 == 0
        mgr._ingest("j1")
        assert mgr.search(STUDY, SearchQuery()).entries[0].catalog.owner == "private-after-withdrawal"
        assert not private_result.exists()
        mgr._reconciler.apply(
            STUDY, "site-a", Selection(None), build_result(ResultStatus.OK, withheld=["clinical-research"]), "t"
        )
        assert mgr.search(STUDY, SearchQuery()).total == 0
        assert b"private-after-withdrawal" not in archive.read_bytes()

    def test_failure_to_save_the_private_result_blocks_archiving_and_keeps_the_source(self, mgr, tmp_path, monkeypatch):
        held(mgr, "j1")
        run_dir = tmp_path / "j1"
        run_dir.mkdir()
        source = run_dir / "catalog_results.json"
        source.write_text("{}")
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(run_dir))

        def fail_save(*args):
            raise OSError("disk full")

        from nvflare.app_common.fdn.catalog.server import artifacts

        monkeypatch.setattr(artifacts, "write_atomically", fail_save)
        ctx = event(mgr, EventType.BEFORE_JOB_WORKSPACE_ARCHIVE, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) and source.exists()


class TestIngest:
    def test_a_manifest_packaged_by_the_real_filesystem_archiver_is_applied(self, mgr, tmp_path):
        publication = manifest(**{"site-a": ok_entry()})
        held(mgr, "j1")
        run_dir = tmp_path / "j1"
        run_dir.mkdir()
        (run_dir / "catalog_results.json").write_text(
            json.dumps({"manifest": publication, "mac": manifest_mac(job_key(KEY, "j1"), "j1", publication)})
        )
        archive = tmp_path / "workspace.zip"
        _write_multi(str(archive), [str(run_dir)])
        mgr.jobs.workspaces["j1"] = archive.read_bytes()
        mgr._ingest("j1")
        assert mgr.search(STUDY, SearchQuery()).total == 1

    def test_missing_workspace_records_failure_and_releases_the_study(self, mgr, monkeypatch):
        held(mgr, "j1", deploy_map={"app": ["site-a"]})

        def missing(*args):
            raise StorageException("workspace absent")

        monkeypatch.setattr(mgr.jobs, "get_storage_component", missing)
        mgr._ingest("j1")
        assert not mgr._in_flight
        assert state(mgr, "site-a")["client"]["reason_code"] == "NO_RESULT"
        # The stored job is terminal, so it cannot block the next read either.
        mgr.jobs.metas["j1"]["status"] = "FINISHED:FAILED_TO_RUN"
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_every_client_of_the_manifest_is_applied_in_the_jobs_study(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ok_entry(), "site-b": ok_entry(display_name="B")}))
        mgr._ingest("j1")
        assert mgr._store.search(STUDY, SearchQuery()).total == 2 and state(mgr, "site-a")["client"]["health"] == "OK"
        assert mgr._in_flight == {}  # the Study is free for the next read

    def test_a_job_without_a_study_is_the_default_study(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}), study=None)
        mgr._ingest("j1")
        assert mgr._store.search("default", SearchQuery()).total == 1

    def test_a_job_that_is_not_a_catalog_job_is_ignored(self, mgr):
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry()}), marker=False)
        mgr._ingest("j1")
        mgr._ingest("gone")
        assert not mgr._store._clients  # nothing stored

    def test_a_manifest_a_job_ships_in_its_own_app_is_not_the_result(self, mgr):
        forged = json.dumps({"manifest": manifest(**{"site-x": ok_entry()}), "mac": "0" * 64})
        held(mgr, "j1", deploy_map={"app": ["site-a"]})
        mgr.jobs.raw["j1"] = {"j1/app_server/config/catalog_results.json": forged}
        mgr._ingest("j1")
        assert set(mgr._store.health(STUDY)[0][k] for k in ("client", "reason_code")) == {"site-a", "NO_RESULT"}

    @pytest.mark.parametrize("signed", ["bad mac", "no mac", "unsigned", "another job"])
    def test_a_manifest_in_the_right_place_without_this_jobs_signature_applies_nothing(self, mgr, signed):
        forged = manifest(**{"site-x": ok_entry()})
        content = {
            "bad mac": {"manifest": forged, "mac": "0" * 64},
            "no mac": {"manifest": forged, "mac": None},
            "unsigned": forged,
            "another job": {"manifest": forged, "mac": manifest_mac(job_key(KEY, "j9"), "j9", forged)},
        }[signed]
        held(mgr, "j1", deploy_map={"app": ["site-a"]})
        mgr.jobs.raw["j1"] = {"j1/catalog_results.json": json.dumps(content)}
        mgr._ingest("j1")
        assert [h["client"] for h in mgr._store.health(STUDY)] == ["site-a"]  # no site-x entry

    def test_without_a_key_nothing_is_applied(self, mgr):
        mgr._key = None
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}), deploy_map={"app": []})
        mgr._ingest("j1")
        assert not mgr._store._clients  # nothing stored

    def test_a_job_without_a_manifest_counts_a_failure_for_each_of_its_sites(self, mgr):
        seed(mgr, client="site-a")
        seed(mgr, client="site-b")
        held(mgr, "j1")  # deployed to @ALL: every site the catalog knows in the Study
        mgr._ingest("j1")
        held(mgr, "j2", deploy_map={"app_server": ["server"], "app_site-a": ["site-a"]})
        mgr._ingest("j2")
        failures = {h["client"]: (h["health"], h["reason_code"], h["failures"]) for h in mgr._store.health(STUDY)}
        assert failures == {"site-a": ("FAILED", "NO_RESULT", 2), "site-b": ("FAILED", "NO_RESULT", 1)}
        assert state(mgr, "site-a")["datasets"]["clinical-research"]["available"] is True

    def test_a_job_that_never_started_counts_no_site_failure_and_leaves_no_context(self, mgr, tmp_path):
        seed(mgr, client="site-a")
        run_dir = tmp_path / "run" / "j1"
        run_dir.mkdir(parents=True)
        context = run_dir / "catalog_context.json"
        context.write_text("{}")  # the controller removes this as soon as it starts, so the job never started
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(run_dir))
        held(mgr, "j1")
        mgr._ingest("j1")
        assert state(mgr, "site-a")["client"]["failures"] == 0 and state(mgr, "site-a")["client"]["health"] == "OK"
        assert not context.exists() and not mgr._in_flight

    def test_a_site_that_failed_to_answer_counts_a_failure_and_keeps_its_catalog(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("j1")
        held(mgr, "j2", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j2")
        row = state(mgr, "site-a")
        assert row["client"]["health"] == "FAILED" and row["client"]["reason_code"] == "TIMEOUT"
        assert row["client"]["failures"] == 1 and "clinical-research" in row["datasets"]

    def test_an_unsupported_result_counts_as_a_failure(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ManifestEntry("OK", {"format_version": 9})}))
        mgr._ingest("j1")
        assert state(mgr, "site-a")["client"]["reason_code"] == "UNSUPPORTED_RESULT"

    def test_an_ok_return_code_without_a_result_records_no_result(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ManifestEntry("OK", None)}))
        mgr._ingest("j1")
        health = state(mgr, "site-a")["client"]
        assert health["health"] == "FAILED" and health["reason_code"] == "NO_RESULT"

    def test_an_aborted_job_reported_twice_is_applied_once(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j1")  # JOB_ABORTED
        mgr._ingest("j1")  # then JOB_COMPLETED
        assert state(mgr, "site-a")["client"]["failures"] == 1

    def test_one_clients_failure_does_not_stop_the_others(self, mgr, monkeypatch):
        held(mgr, "j1", manifest(**{"site-a": ok_entry(), "site-b": ok_entry()}))
        real = mgr._reconciler.apply

        def apply(study, client, *args):
            if client == "site-a":
                raise RuntimeError("disk full")
            return real(study, client, *args)

        monkeypatch.setattr(mgr._reconciler, "apply", apply)
        mgr._ingest("j1")
        assert state(mgr, "site-b")["client"]["health"] == "OK" and mgr._in_flight == {}


class TestEvents:
    @pytest.mark.parametrize("event_type", [EventType.JOB_COMPLETED, EventType.JOB_ABORTED])
    def test_a_finished_job_is_queued_without_doing_any_work_on_the_event_thread(self, mgr, event_type):
        event(mgr, event_type, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert mgr._finished.get_nowait() == "j1"

    def test_the_system_end_event_stops_the_manager(self, mgr):
        mgr.handle_event(EventType.SYSTEM_END, FLContext())
        assert mgr._stop.is_set()


class TestDownloads:
    @pytest.mark.parametrize("status", ["SUBMITTED", "FINISHED:ABORTED", "FINISHED:FAILED_TO_RUN"])
    def test_missing_workspace_does_not_block_definition_download(self, mgr, monkeypatch, status):
        mgr.jobs.add("j1", status=status)

        def missing(*args):
            raise StorageException("workspace absent")

        monkeypatch.setattr(mgr.jobs, "get_storage_component", missing)
        ctx = event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_an_existing_workspace_that_cannot_be_read_still_blocks_download(self, mgr, monkeypatch):
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry()}))

        def denied(*args):
            raise StorageException("read denied")

        monkeypatch.setattr(mgr.jobs, "get_storage_component", denied)
        ctx = event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    def test_concurrent_legacy_download_and_ingestion_apply_once_and_leave_no_private_copy(self, mgr, monkeypatch):
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}), status="FINISHED:COMPLETED")
        entered, release = threading.Event(), threading.Event()
        observed = ObservedLock()
        monkeypatch.setattr(mgr, "_result_lock", observed)
        capture = mgr._artifacts.capture_archive
        applications, download_contexts = [], []
        apply = mgr._reconciler.apply

        def gated_capture(*args):
            capture(*args)
            entered.set()
            release.wait()

        def record_apply(*args):
            applications.append(args[1])
            return apply(*args)

        def download():
            download_contexts.append(event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"}))

        monkeypatch.setattr(mgr._artifacts, "capture_archive", gated_capture)
        monkeypatch.setattr(mgr._reconciler, "apply", record_apply)
        with ThreadGroup(release) as threads:
            threads.start(download)
            assert entered.wait(3.0), "download never captured the legacy result"
            threads.start(mgr._ingest, "j1")
            assert observed.blocked.wait(3.0), "ingestion did not wait for the archive replacement"
            assert not applications
            release.set()
        assert download_contexts[0].get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert applications == ["site-a"] and not mgr._in_flight
        assert mgr.search(STUDY, SearchQuery()).total == 1
        with pytest.raises(FileNotFoundError):
            mgr._artifacts.read("j1", KEY)
        with zipfile.ZipFile(io.BytesIO(mgr.jobs.workspaces["j1"])) as zf:
            assert not zf.namelist()

    @pytest.mark.parametrize("prefix", ["", "j1/"])
    def test_legacy_archives_hide_publications_and_context_but_preserve_logs(self, mgr, prefix):
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry(owner="old-owner")}), status="FINISHED:COMPLETED")
        if prefix:
            publication = mgr.jobs.manifests.pop("j1")
            mgr.jobs.raw["j1"] = {
                prefix
                + "catalog_results.json": json.dumps(
                    {"manifest": publication, "mac": manifest_mac(job_key(KEY, "j1"), "j1", publication)}
                )
            }
        mgr.jobs.raw.setdefault("j1", {}).update(
            {
                prefix + "catalog_context.json": "secret",
                prefix + "catalog_results.json.tmp": "partial metadata",
                "log.txt": "safe",
            }
        )
        ctx = event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        with zipfile.ZipFile(io.BytesIO(mgr.jobs.workspaces["j1"])) as zf:
            assert zf.namelist() == ["log.txt"] and zf.read("log.txt") == b"safe"

    def test_download_before_ingestion_preserves_the_private_result(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}), status="FINISHED:COMPLETED")
        ctx = event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        mgr._ingest("j1")
        assert mgr.search(STUDY, SearchQuery()).total == 1

    def test_an_archive_that_cannot_be_rewritten_blocks_the_download(self, mgr, monkeypatch):
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry()}), status="FINISHED:COMPLETED")

        def denied(*args):
            raise OSError("store unavailable")

        monkeypatch.setattr(mgr.jobs, "save_workspace", denied)
        ctx = event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    def test_other_jobs_are_left_unchanged(self, mgr):
        mgr.jobs.add("j1", marker=False)
        event(mgr, EventType.BEFORE_JOB_DOWNLOAD, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert not mgr.jobs.workspaces


class TestRecoveryLoop:
    @pytest.mark.parametrize("completion_events", [False, True])
    def test_recovery_sweeps_are_spaced_out_without_delaying_completion_events(
        self, mgr, monkeypatch, completion_events
    ):
        clock = ManualClock(now=0)
        monkeypatch.setattr(manager_module, "time", clock)
        recovered, applied = [], []

        class Polls:
            def get(self, timeout):
                assert timeout == 1.0
                clock.advance(1)
                if clock.monotonic() >= 65:
                    mgr._stop.set()
                if completion_events and clock.monotonic() % 2 == 0:
                    return "finished"
                raise queue.Empty

        mgr._finished = Polls()
        monkeypatch.setattr(mgr, "_recover_finished", lambda: recovered.append(clock.monotonic()))
        monkeypatch.setattr(mgr, "_ingest", lambda job_id: applied.append((job_id, clock.monotonic())))
        mgr._run()
        assert recovered == [30, 60]
        assert applied == ([("finished", second) for second in range(2, 65, 2)] if completion_events else [])


class TestNotReady:
    def test_queries_before_the_service_has_started_say_so(self, mgr):
        mgr._store = None
        with pytest.raises(NotReady):
            mgr.search(STUDY, SearchQuery())
        with pytest.raises(NotReady):
            mgr.health(STUDY)


class TestHealth:
    def test_a_site_is_stale_when_its_last_success_is_older_than_the_limit(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("j1")
        assert [h["stale"] for h in mgr.health(STUDY)] == [False]
        mgr._now = lambda: datetime(2026, 6, 1, 2, tzinfo=timezone.utc)
        assert [h["stale"] for h in mgr.health(STUDY)] == [True]

    def test_a_site_that_never_succeeded_is_stale(self, mgr):
        held(mgr, "j1", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j1")
        assert mgr.health(STUDY)[0]["stale"] is True


def test_context_write_failure_vetoes_launch_and_preserves_site_health(mgr, monkeypatch):
    from nvflare.apis.fl_constant import ReservedKey
    from nvflare.private.fed.utils.fed_utils import get_job_launcher

    seed(mgr)
    before = mgr._store.client_state(STUDY, "site-a")
    mgr._workspace = SimpleNamespace(get_run_dir=lambda _: "/unused")

    def denied(*args):
        raise OSError("disk full")

    def fire(event_type, ctx):
        ctx.set_prop(FLContextKey.JOB_LAUNCHER, [object()], private=True, sticky=False)
        mgr.handle_event(event_type, ctx)

    monkeypatch.setattr(manager_module, "write_atomically", denied)
    mgr._engine.new_context = FLContext
    mgr._engine.fire_event = fire
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, mgr._engine, private=True, sticky=False)
    with pytest.raises(RuntimeError, match="Job launch blocked: catalog job context"):
        get_job_launcher(catalog_meta("j1"), ctx)
    assert not mgr._in_flight
    mgr.jobs.add("j1", status="FINISHED:FAILED_TO_RUN")
    mgr._ingest("j1")
    assert mgr._store.client_state(STUDY, "site-a") == before
