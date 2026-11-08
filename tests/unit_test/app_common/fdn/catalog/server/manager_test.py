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
from tests.timing_utils import ManualClock
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
    config_file = tmp_path / "config_fed_server.json"
    config_file.write_text(
        json.dumps(
            {
                "format_version": 2,
                "workflows": [
                    {"id": "controller", "path": "nvflare.app_common.fdn.catalog.server.controller.CatalogController"}
                ],
            }
        )
    )
    manager._workspace = SimpleNamespace(
        get_root_dir=lambda: str(tmp_path),
        get_run_dir=lambda job_id: str(tmp_path / job_id),
        get_server_app_config_file_path=lambda _: str(config_file),
        get_app_custom_dir=lambda _: str(tmp_path / "server_custom"),
    )
    manager._reconciler = Reconciler(manager._store, LIMITS)
    manager._now = lambda: T0
    manager._key = KEY
    manager.jobs = jobs
    yield manager
    manager._stop.set()
    if manager._thread is not None:
        manager._thread.join(timeout=3.0)
        assert not manager._thread.is_alive()


def pending(mgr, job_id, manifest=None, study=STUDY, **kw):
    """A launched catalog job with a result waiting to be processed."""
    mgr.jobs.add(job_id, manifest, study=study, **kw)
    mgr._pending.add(job_id)


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


class TestOverlappingReads:
    @pytest.fixture
    def run_dirs(self, mgr, tmp_path):
        mgr._workspace.get_run_dir = lambda job_id: str(tmp_path / "run" / job_id)
        return lambda job_id: (tmp_path / "run" / job_id).mkdir(parents=True) or tmp_path / "run" / job_id

    @pytest.mark.parametrize("status", ["SUBMITTED", "RUNNING", "FINISHED:COMPLETED"])
    def test_another_catalog_job_never_blocks_submission(self, mgr, status):
        pending(mgr, "j1", status=status)
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_overlapping_jobs_get_independent_contexts_for_the_same_study(self, mgr, run_dirs):
        first, second = run_dirs("j1"), run_dirs("j2")
        for job_id in ("j1", "j2"):
            ctx = event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta(job_id)})
            assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert (first / "catalog_context.json").exists() and (second / "catalog_context.json").exists()
        assert mgr._pending == {"j1", "j2"}
        assert (
            json.loads((first / "catalog_context.json").read_text())["key"]
            != json.loads((second / "catalog_context.json").read_text())["key"]
        )

    def test_last_applied_result_wins_even_when_its_job_was_submitted_first(self, mgr):
        pending(mgr, "first", manifest(**{"site-a": ok_entry(display_name="First job")}))
        pending(mgr, "second", manifest(**{"site-a": ok_entry(display_name="Second job")}))
        mgr.jobs.metas["first"]["submit_time"] = 1
        mgr.jobs.metas["second"]["submit_time"] = 2
        mgr._ingest("second")
        assert state(mgr, "site-a")["datasets"]["clinical-research"]["metadata"]["display_name"] == "Second job"
        mgr._ingest("first")
        row = state(mgr, "site-a")["datasets"]["clinical-research"]
        assert row["metadata"]["display_name"] == "First job" and row["catalog_version"] == 2
        assert not mgr._pending

    def test_last_explicit_withdrawal_or_publication_wins(self, mgr):
        seed(mgr)
        pending(
            mgr,
            "withdraw",
            manifest(**{"site-a": ManifestEntry("OK", build_result(ResultStatus.OK, withheld=["clinical-research"]))}),
        )
        pending(mgr, "publish", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("withdraw")
        assert mgr.search(STUDY, SearchQuery()).total == 0
        mgr._ingest("publish")
        assert mgr.search(STUDY, SearchQuery()).total == 1
        pending(
            mgr,
            "withdraw-again",
            manifest(**{"site-a": ManifestEntry("OK", build_result(ResultStatus.OK, withheld=["clinical-research"]))}),
        )
        mgr._ingest("withdraw-again")
        assert mgr.search(STUDY, SearchQuery()).total == 0

    def test_partial_and_empty_results_leave_unmentioned_datasets_unchanged(self, mgr):
        pending(mgr, "a", manifest(**{"site-a": ok_entry(dataset="a")}))
        pending(mgr, "b", manifest(**{"site-a": ok_entry(dataset="b")}))
        pending(mgr, "empty", manifest(**{"site-a": ManifestEntry("OK", build_result(ResultStatus.OK))}))
        mgr._ingest("b")
        mgr._ingest("a")
        before = state(mgr, "site-a")["datasets"]
        mgr._ingest("empty")
        assert set(before) == {"a", "b"} and state(mgr, "site-a")["datasets"] == before

    def test_failed_application_does_not_block_another_read_and_retry_applies_last(self, mgr, run_dirs, monkeypatch):
        pending(
            mgr, "retry", manifest(**{"site-a": ok_entry(display_name="Retried update")}), status="FINISHED:COMPLETED"
        )
        commit = mgr._store.commit

        def unavailable(*args):
            raise OSError("temporary storage failure")

        monkeypatch.setattr(mgr._store, "commit", unavailable)
        mgr._ingest("retry")
        assert mgr._pending == {"retry"}
        monkeypatch.setattr(mgr._store, "commit", commit)
        run_dirs("next")
        mgr.jobs.add("next", manifest(**{"site-a": ok_entry(display_name="Other update")}))
        submitted = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        launched = event(mgr, EventType.BEFORE_JOB_LAUNCH, **{FLContextKey.JOB_META: catalog_meta("next")})
        assert submitted.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert launched.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        mgr._ingest("next")
        assert state(mgr, "site-a")["datasets"]["clinical-research"]["metadata"]["display_name"] == "Other update"
        mgr._recover_finished()
        assert state(mgr, "site-a")["datasets"]["clinical-research"]["metadata"]["display_name"] == "Retried update"
        assert not mgr._pending

    def test_an_untracked_jobs_result_is_not_applied(self, mgr):
        pending(mgr, "j1")
        mgr.jobs.add("untracked", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("untracked")
        assert not mgr._store._clients and mgr._pending == {"j1"}

    def test_restart_tracks_all_running_jobs_of_the_same_study(self, mgr, tmp_path):
        mgr.jobs.add("j1", status="RUNNING")
        mgr.jobs.add("j2", status="RUNNING")
        mgr.jobs.add("waiting", status="SUBMITTED")
        mgr.jobs.add("done", status="FINISHED:COMPLETED")
        ctx = SimpleNamespace(
            get_engine=lambda: mgr._engine, get_workspace=lambda: SimpleNamespace(get_root_dir=lambda: str(tmp_path))
        )
        mgr._start(ctx)
        mgr._stop.set()
        assert mgr._pending == {"j1", "j2"}

    @pytest.mark.parametrize("status", ["FINISHED:ABORTED", "FINISHED:FAILED_TO_RUN"])
    @pytest.mark.parametrize("gone", [False, True])
    def test_a_terminal_or_deleted_job_is_cleaned_up_without_an_event(self, mgr, gone, status):
        pending(mgr, "j1", status=status)
        if gone:
            del mgr.jobs.metas["j1"]
        mgr._recover_finished()
        assert not mgr._pending


class TestHandOver:
    """The server puts the job's key and the names it holds in the job's run dir: a job process in a container
    sees its own job directory and nothing of the server's catalog store."""

    def launch(self, mgr, tmp_path, meta):
        run_dir = str(tmp_path / "run" / "j1")
        os.makedirs(run_dir, exist_ok=True)
        mgr._workspace.get_run_dir = lambda job_id: run_dir
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

    @pytest.mark.parametrize("change", ["byoc", "custom", "workflow", "component", "filter", "args", "nested-id"])
    def test_unapproved_server_code_gets_neither_key_nor_indexed_names(self, mgr, tmp_path, change):
        from nvflare.apis.app_validation import AppValidationKey

        seed(mgr)
        config_path = Path(mgr._workspace.get_server_app_config_file_path("j1"))
        config = json.loads(config_path.read_text())
        meta = catalog_meta("j1")
        if change == "byoc":
            meta[AppValidationKey.BYOC] = True
        elif change == "custom":
            custom = Path(mgr._workspace.get_app_custom_dir("j1"))
            custom.mkdir()
            (custom / "controller.py").write_text("# job-supplied code")
        elif change == "workflow":
            config["workflows"][0]["path"] = "nvflare.app_common.workflows.statistics_controller.StatisticsController"
        elif change == "component":
            config["components"] = [{"id": "custom", "path": "untrusted.CustomComponent"}]
        elif change == "filter":
            config["task_result_filters"] = [{"path": "untrusted.CustomFilter"}]
        elif change == "nested-id":
            config["workflows"][0]["id"] = {"path": "untrusted.CustomComponent"}
        else:
            config["workflows"][0]["args"] = {"unknown": "argument"}
        config_path.write_text(json.dumps(config))
        path = self.launch(mgr, tmp_path, meta)
        assert not Path(path).exists() and not mgr._pending

    @pytest.mark.parametrize("mode", ["docker", "k8s", "slurm"])
    @pytest.mark.parametrize("source", ["server", "default", "legacy"])
    def test_job_image_is_rejected_even_without_a_byoc_flag(self, mgr, tmp_path, mode, source):
        meta = catalog_meta("j1")
        if source == "legacy":
            meta["resource_spec"] = {"server": {mode: {"image": "untrusted/catalog:latest"}}}
        else:
            meta["launcher_spec"] = {source: {mode: {"image": "untrusted/catalog:latest"}}}
        path = self.launch(mgr, tmp_path, meta)
        assert not Path(path).exists() and not mgr._pending

    def test_client_image_and_portable_server_resources_do_not_override_server_execution(self, mgr, tmp_path):
        meta = catalog_meta("j1")
        meta["launcher_spec"] = {"site-a": {"docker": {"image": "site-approved/client:latest"}}}
        meta["resource_spec"] = {"server": {"num_of_cpus": 2}}
        assert Path(self.launch(mgr, tmp_path, meta)).exists()

    def test_recipe_with_log_streaming_passes_the_server_gate(self, mgr, tmp_path):
        from nvflare.recipe.catalog import CatalogRecipe

        recipe = CatalogRecipe(sites=["site-a"])
        recipe.enable_log_streaming("log.txt")
        recipe.export(str(tmp_path / "export"))
        mgr._workspace.get_server_app_config_file_path = lambda _: str(
            tmp_path / "export/catalog_read/app_server/config/config_fed_server.json"
        )
        path = self.launch(mgr, tmp_path, catalog_meta("j1"))
        assert Path(path).exists() and mgr._pending == {"j1"}


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

    def test_a_service_that_did_not_start_does_not_block_other_jobs_from_archiving(self, mgr, tmp_path):
        mgr._artifacts = None
        run_dir, ctx = self.archive(mgr, tmp_path, "keep.json")
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_reserved_private_files_are_removed_even_without_the_catalog_marker(self, mgr, tmp_path):
        run_dir, ctx = self.archive(mgr, tmp_path, "catalog_context.json", marker=False)
        assert not (run_dir / "catalog_context.json").exists() and ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_a_context_that_cannot_be_removed_blocks_the_archive(self, mgr, tmp_path, monkeypatch):
        def denied(path, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "unlink", denied)
        _, ctx = self.archive(mgr, tmp_path, "catalog_context.json")
        assert "could not be secured" in ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)

    def test_result_is_captured_privately_and_never_enters_the_downloadable_archive(self, mgr, tmp_path, monkeypatch):
        publication = manifest(**{"site-a": ok_entry(owner="private-after-withdrawal")})
        pending(mgr, "j1")
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
        pending(mgr, "j1")
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
        pending(mgr, "j1")
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

    def test_missing_workspace_finishes_processing_without_blaming_a_site(self, mgr, monkeypatch):
        pending(mgr, "j1", deploy_map={"app": ["site-a"]})

        def missing(*args):
            raise StorageException("workspace absent")

        monkeypatch.setattr(mgr.jobs, "get_storage_component", missing)
        mgr._ingest("j1")
        assert not mgr._pending
        assert not mgr._store._clients  # the job failed; nothing says which site did
        # A failed job does not prevent another submission.
        mgr.jobs.metas["j1"]["status"] = "FINISHED:FAILED_TO_RUN"
        ctx = event(mgr, EventType.SUBMIT_JOB, **{FLContextKey.JOB_META: catalog_meta(None)})
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    def test_every_client_of_the_manifest_is_applied_in_the_jobs_study(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ok_entry(), "site-b": ok_entry(display_name="B")}))
        mgr._ingest("j1")
        assert mgr._store.search(STUDY, SearchQuery()).total == 2 and state(mgr, "site-a")["client"]["health"] == "OK"
        assert mgr._pending == set()  # this job no longer needs processing

    def test_a_job_without_a_study_is_the_default_study(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}), study=None)
        mgr._ingest("j1")
        assert mgr._store.search("default", SearchQuery()).total == 1

    def test_a_job_that_is_not_a_catalog_job_is_ignored(self, mgr):
        mgr.jobs.add("j1", manifest(**{"site-a": ok_entry()}), marker=False)
        mgr._ingest("j1")
        mgr._ingest("gone")
        assert not mgr._store._clients  # nothing stored

    def test_a_manifest_a_job_ships_in_its_own_app_is_not_the_result(self, mgr):
        forged = json.dumps({"manifest": manifest(**{"site-x": ok_entry()}), "mac": "0" * 64})
        pending(mgr, "j1", deploy_map={"app": ["site-a"]})
        mgr.jobs.raw["j1"] = {"j1/app_server/config/catalog_results.json": forged}
        mgr._ingest("j1")
        assert not mgr._store._clients and not mgr._pending

    @pytest.mark.parametrize("signed", ["bad mac", "no mac", "unsigned", "another job"])
    def test_a_manifest_in_the_right_place_without_this_jobs_signature_applies_nothing(self, mgr, signed):
        forged = manifest(**{"site-x": ok_entry()})
        content = {
            "bad mac": {"manifest": forged, "mac": "0" * 64},
            "no mac": {"manifest": forged, "mac": None},
            "unsigned": forged,
            "another job": {"manifest": forged, "mac": manifest_mac(job_key(KEY, "j9"), "j9", forged)},
        }[signed]
        pending(mgr, "j1", deploy_map={"app": ["site-a"]})
        mgr.jobs.raw["j1"] = {"j1/catalog_results.json": json.dumps(content)}
        mgr._ingest("j1")
        assert not mgr._store._clients and not mgr._pending  # nothing applied, nobody blamed

    def test_without_a_key_nothing_is_applied(self, mgr):
        mgr._key = None
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}), deploy_map={"app": []})
        mgr._ingest("j1")
        assert not mgr._store._clients  # nothing stored

    def test_a_job_without_a_manifest_changes_no_sites_health(self, mgr):
        seed(mgr, client="site-a")
        seed(mgr, client="site-b")
        pending(mgr, "j1")  # deployed to @ALL: every site the catalog knows in the Study
        mgr._ingest("j1")
        pending(mgr, "j2", deploy_map={"app_server": ["server"], "app_site-a": ["site-a"]})
        mgr._ingest("j2")
        health = {h["client"]: (h["health"], h["failures"]) for h in mgr._store.health(STUDY)}
        assert health == {"site-a": ("OK", 0), "site-b": ("OK", 0)}
        assert state(mgr, "site-a")["datasets"]["clinical-research"]["available"] is True

    def test_a_job_that_left_no_result_leaves_no_context_file(self, mgr, tmp_path):
        seed(mgr, client="site-a")
        run_dir = tmp_path / "run" / "j1"
        run_dir.mkdir(parents=True)
        context = run_dir / "catalog_context.json"
        context.write_text("{}")  # the controller removes this as soon as it starts, so this job never did
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(run_dir))
        pending(mgr, "j1")
        mgr._ingest("j1")
        assert state(mgr, "site-a")["client"]["failures"] == 0 and state(mgr, "site-a")["client"]["health"] == "OK"
        assert not context.exists() and not mgr._pending

    def test_a_failed_cleanup_does_not_skip_the_other_or_keep_the_study(self, mgr, tmp_path, monkeypatch):
        run_dir = tmp_path / "run" / "j1"
        run_dir.mkdir(parents=True)
        context = run_dir / "catalog_context.json"
        context.write_text("{}")
        mgr._workspace = SimpleNamespace(get_run_dir=lambda job_id: str(run_dir))
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}))

        def broken(job_id):
            raise OSError("disk full")

        monkeypatch.setattr(mgr._artifacts, "discard", broken)
        mgr._ingest("j1")
        assert not context.exists() and mgr._pending == {"j1"}
        assert mgr._store.search(STUDY, SearchQuery()).total == 1  # the result was applied before cleanup

    def test_a_site_that_failed_to_answer_counts_a_failure_and_keeps_its_catalog(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("j1")
        pending(mgr, "j2", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j2")
        row = state(mgr, "site-a")
        assert row["client"]["health"] == "FAILED" and row["client"]["reason_code"] == "TIMEOUT"
        assert row["client"]["failures"] == 1 and "clinical-research" in row["datasets"]

    def test_an_unsupported_result_counts_as_a_failure(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ManifestEntry("OK", {"format_version": 9})}))
        mgr._ingest("j1")
        assert state(mgr, "site-a")["client"]["reason_code"] == "UNSUPPORTED_RESULT"

    def test_an_ok_return_code_without_a_result_records_no_result(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ManifestEntry("OK", None)}))
        mgr._ingest("j1")
        health = state(mgr, "site-a")["client"]
        assert health["health"] == "FAILED" and health["reason_code"] == "NO_RESULT"

    def test_an_aborted_job_reported_twice_is_applied_once(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j1")  # JOB_ABORTED
        mgr._ingest("j1")  # then JOB_COMPLETED
        assert state(mgr, "site-a")["client"]["failures"] == 1

    def test_one_clients_failure_does_not_stop_the_others(self, mgr, monkeypatch):
        pending(mgr, "j1", manifest(**{"site-a": ok_entry(), "site-b": ok_entry()}))
        real = mgr._reconciler.apply

        def apply(study, client, *args, **kwargs):
            if client == "site-a":
                raise RuntimeError("disk full")
            return real(study, client, *args, **kwargs)

        monkeypatch.setattr(mgr._reconciler, "apply", apply)
        mgr._ingest("j1")
        assert state(mgr, "site-b")["client"]["health"] == "OK" and mgr._pending == set()


class TestEvents:
    @pytest.mark.parametrize("event_type", [EventType.JOB_COMPLETED, EventType.JOB_ABORTED])
    def test_a_finished_job_is_queued_without_doing_any_work_on_the_event_thread(self, mgr, event_type):
        event(mgr, event_type, **{FLContextKey.CURRENT_JOB_ID: "j1"})
        assert mgr._finished.get_nowait() == "j1"

    def test_the_system_end_event_stops_the_manager(self, mgr):
        mgr.handle_event(EventType.SYSTEM_END, FLContext())
        assert mgr._stop.is_set()


class TestDownloads:
    def staged(self, tmp_path, entries):
        path = tmp_path / "workspace.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        return path

    def prepare(self, mgr, path=None):
        return event(
            mgr,
            EventType.BEFORE_JOB_DOWNLOAD,
            **{
                FLContextKey.CURRENT_JOB_ID: "j1",
                FLContextKey.JOB_DOWNLOAD_FILES: {} if path is None else {"workspace": str(path)},
            },
        )

    def test_definition_only_download_has_no_workspace_to_scrub(self, mgr):
        assert self.prepare(mgr).get_prop(FLContextKey.JOB_BLOCK_REASON) is None

    @pytest.mark.parametrize("prefix", ["", "j1/", "j1/app_server/"])
    def test_unmarked_unsigned_archives_hide_private_files_without_modifying_store(self, mgr, tmp_path, prefix):
        path = self.staged(
            tmp_path,
            {
                prefix + "catalog_results.json": '{"unsigned": "site-private-metadata"}',
                prefix + "catalog_context.json": "key",
                prefix + "catalog_results.json.tmp": "partial",
                "log.txt": "safe log",
            },
        )
        stored = path.read_bytes()
        mgr.jobs.workspaces["j1"] = stored
        ctx = self.prepare(mgr, path)
        assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert mgr.jobs.workspaces["j1"] == stored
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist() == ["log.txt"] and archive.read("log.txt") == b"safe log"

    def test_redacting_a_download_does_not_consume_the_pending_result(self, mgr, tmp_path):
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}), status="FINISHED:COMPLETED")
        path = tmp_path / "workspace.zip"
        path.write_bytes(mgr.jobs.get_storage_component("j1", "workspace", None))
        assert self.prepare(mgr, path).get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        mgr._ingest("j1")
        assert mgr.search(STUDY, SearchQuery()).total == 1 and not mgr._pending
        with zipfile.ZipFile(path) as archive:
            assert not archive.namelist()

    def test_normal_model_archive_is_not_read_or_rewritten(self, mgr, tmp_path, monkeypatch):
        path = self.staged(tmp_path, {"model.pt": "model bytes"})
        before = path.read_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError("normal archive member content was read")

        monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
        assert self.prepare(mgr, path).get_prop(FLContextKey.JOB_BLOCK_REASON) is None
        assert path.read_bytes() == before

    def test_redaction_failure_blocks_transfer_and_leaves_no_partial_copy(self, mgr, tmp_path, monkeypatch):
        from nvflare.app_common.fdn.catalog.server import artifacts

        path = self.staged(tmp_path, {"catalog_results.json": "private", "log.txt": "safe"})
        before = path.read_bytes()

        def full(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(artifacts.shutil, "copyfileobj", full)
        assert self.prepare(mgr, path).get_prop(FLContextKey.JOB_BLOCK_REASON)
        assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [
            "config_fed_server.json",
            "workspace.zip",
        ]


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
        pending(mgr, "j1", manifest(**{"site-a": ok_entry()}))
        mgr._ingest("j1")
        assert [h["stale"] for h in mgr.health(STUDY)] == [False]
        mgr._now = lambda: datetime(2026, 6, 1, 2, tzinfo=timezone.utc)
        assert [h["stale"] for h in mgr.health(STUDY)] == [True]

    def test_a_site_that_never_succeeded_is_stale(self, mgr):
        pending(mgr, "j1", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}))
        mgr._ingest("j1")
        assert mgr.health(STUDY)[0]["stale"] is True


def test_context_write_failure_vetoes_only_that_launch_and_preserves_site_health(mgr, monkeypatch):
    from nvflare.apis.fl_constant import ReservedKey
    from nvflare.private.fed.utils.fed_utils import get_job_launcher

    seed(mgr)
    before = mgr._store.client_state(STUDY, "site-a")
    mgr._workspace.get_run_dir = lambda _: "/unused"

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
    assert not mgr._pending
    mgr.jobs.add("j1", status="FINISHED:FAILED_TO_RUN")
    mgr._ingest("j1")
    assert mgr._store.client_state(STUDY, "site-a") == before


def test_failed_commit_retains_result_and_retries_only_unapplied_clients(mgr, tmp_path, monkeypatch):
    seed(mgr)
    publication = manifest(
        **{
            "site-a": ManifestEntry("OK", build_result(ResultStatus.OK, withheld=["clinical-research"])),
            "site-b": ManifestEntry("TIMEOUT", None),
        }
    )
    pending(mgr, "j1", status="FINISHED:COMPLETED")
    run_dir = tmp_path / "j1"
    run_dir.mkdir()
    (run_dir / "catalog_results.json").write_text(
        json.dumps({"manifest": publication, "mac": manifest_mac(job_key(KEY, "j1"), "j1", publication)})
    )
    mgr._artifacts.capture("j1", str(run_dir), True)
    inbox = tmp_path / "fdn/catalog-results/j1.json"
    original = mgr._store.commit
    blocked = True

    def commit(study, client, state):
        if client == "site-a" and blocked:
            raise OSError("temporary storage failure")
        return original(study, client, state)

    monkeypatch.setattr(mgr._store, "commit", commit)
    mgr._ingest("j1")
    assert inbox.exists() and mgr._pending == {"j1"}
    assert state(mgr, "site-b")["client"]["failures"] == 1
    blocked = False
    mgr._recover_finished()
    assert mgr.search(STUDY, SearchQuery()).total == 0
    assert state(mgr, "site-a")["datasets"]["clinical-research"]["withheld"]
    assert state(mgr, "site-b")["client"]["failures"] == 1
    assert not inbox.exists() and not mgr._pending


def test_unreadable_private_result_is_retried_after_storage_recovers(mgr, monkeypatch):
    pending(mgr, "j1", manifest(**{"site-a": ok_entry()}), status="FINISHED:COMPLETED")
    original = mgr._artifacts.read

    def denied(*args):
        raise PermissionError("temporary read failure")

    monkeypatch.setattr(mgr._artifacts, "read", denied)
    mgr._ingest("j1")
    assert mgr._pending == {"j1"}
    monkeypatch.setattr(mgr._artifacts, "read", original)
    mgr._recover_finished()
    assert mgr.search(STUDY, SearchQuery()).total == 1
    assert not mgr._pending


@pytest.mark.parametrize("event_type", [EventType.BEFORE_JOB_DOWNLOAD, EventType.BEFORE_JOB_WORKSPACE_ARCHIVE])
def test_ordinary_job_uses_supplied_metadata_without_an_extra_store_lookup(mgr, monkeypatch, event_type):
    def unavailable(*args):
        raise OSError("job store unavailable after metadata was obtained")

    monkeypatch.setattr(mgr, "_job_meta", unavailable)
    ctx = event(
        mgr,
        event_type,
        **{
            FLContextKey.CURRENT_JOB_ID: "ordinary",
            FLContextKey.JOB_META: {"job_id": "ordinary", "study": STUDY},
        },
    )
    assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None


def test_a_download_of_a_job_with_no_stored_workspace_is_not_refused(mgr, tmp_path):
    ctx = event(
        mgr,
        EventType.BEFORE_JOB_DOWNLOAD,
        **{
            FLContextKey.CURRENT_JOB_ID: "aborted",
            FLContextKey.JOB_META: {"job_id": "aborted", "study": STUDY},
            FLContextKey.JOB_DOWNLOAD_FILES: {"workspace": str(tmp_path / "aborted" / "workspace.zip")},
        },
    )
    assert ctx.get_prop(FLContextKey.JOB_BLOCK_REASON) is None


def test_restart_applies_retained_terminal_result_and_skips_committed_clients(mgr, tmp_path, monkeypatch):
    seed(mgr)
    pending(
        mgr,
        "j1",
        manifest(
            **{
                "site-a": ManifestEntry("OK", build_result(ResultStatus.OK, withheld=["clinical-research"])),
                "site-b": ManifestEntry("TIMEOUT", None),
            }
        ),
        status="FINISHED:COMPLETED",
    )
    commit = mgr._store.commit

    def unavailable(study, client, state):
        if client == "site-a":
            raise OSError("transient failure")
        commit(study, client, state)

    monkeypatch.setattr(mgr._store, "commit", unavailable)
    mgr._ingest("j1")
    assert state(mgr, "site-b")["applied_jobs"] == ["j1"]
    assert state(mgr, "site-b")["client"]["failures"] == 1
    monkeypatch.setattr(manager_module, "manifest_key", lambda _: KEY)
    restarted = CatalogManager(register_commands=False)
    try:
        restarted._start(SimpleNamespace(get_engine=lambda: mgr._engine, get_workspace=lambda: mgr._workspace))
        assert restarted._pending == {"j1"}
        restarted._recover_finished()
        assert restarted.search(STUDY, SearchQuery()).total == 0
        assert state(restarted, "site-b")["client"]["failures"] == 1
        assert not restarted._pending and not restarted._artifacts.pending_jobs()
    finally:
        restarted._stop.set()
        if restarted._thread is not None:
            restarted._thread.join(timeout=3.0)
            assert not restarted._thread.is_alive()


def test_job_cannot_supply_parent_application_progress(mgr, tmp_path):
    pending(mgr, "j1", status="FINISHED:COMPLETED")
    publication = manifest(**{"site-a": ok_entry()})
    run_dir = tmp_path / "j1"
    run_dir.mkdir()
    (run_dir / "catalog_results.json").write_text(
        json.dumps(
            {
                "manifest": publication,
                "mac": manifest_mac(job_key(KEY, "j1"), "j1", publication),
                "applied_clients": ["site-a"],
            }
        )
    )
    mgr._artifacts.capture("j1", str(run_dir), True)
    mgr._ingest("j1")
    assert mgr.search(STUDY, SearchQuery()).total == 1


def test_crash_after_state_commit_is_idempotent_on_restart(mgr, tmp_path, monkeypatch):
    pending(mgr, "j1", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}), status="FINISHED:COMPLETED")
    commit = mgr._store.commit

    def committed_then_lost_ack(*args):
        commit(*args)
        raise OSError("crash after durable state replacement")

    monkeypatch.setattr(mgr._store, "commit", committed_then_lost_ack)
    mgr._ingest("j1")
    assert state(mgr, "site-a")["client"]["failures"] == 1
    assert state(mgr, "site-a")["applied_jobs"] == ["j1"]
    monkeypatch.setattr(manager_module, "manifest_key", lambda _: KEY)
    restarted = CatalogManager(register_commands=False)
    try:
        restarted._start(SimpleNamespace(get_engine=lambda: mgr._engine, get_workspace=lambda: mgr._workspace))
        restarted._recover_finished()
        assert state(restarted, "site-a")["client"]["failures"] == 1
        assert not restarted._pending and not restarted._artifacts.pending_jobs()
    finally:
        restarted._stop.set()
        if restarted._thread is not None:
            restarted._thread.join(timeout=3)
            assert not restarted._thread.is_alive()


def test_receipt_survives_an_overlapping_update_before_retry(mgr, monkeypatch):
    pending(mgr, "old", manifest(**{"site-a": ManifestEntry("TIMEOUT", None)}), status="FINISHED:COMPLETED")
    commit = mgr._store.commit

    def committed_then_lost_ack(*args):
        commit(*args)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(mgr._store, "commit", committed_then_lost_ack)
    mgr._ingest("old")
    monkeypatch.setattr(mgr._store, "commit", commit)
    pending(mgr, "new", manifest(**{"site-a": ok_entry(display_name="latest")}))
    mgr._ingest("new")
    # Lose all process-local job state, as on restart, then replay the old inbox.
    mgr._store = CatalogStore(mgr._workspace.get_root_dir())
    mgr._reconciler = Reconciler(mgr._store, LIMITS)
    mgr._recover_finished()
    assert state(mgr, "site-a")["client"]["failures"] == 0
    assert state(mgr, "site-a")["datasets"]["clinical-research"]["metadata"]["display_name"] == "latest"
    pending(mgr, "next", manifest(**{"site-a": ok_entry()}))
    mgr._ingest("next")
    assert state(mgr, "site-a")["applied_jobs"] == ["next"]
