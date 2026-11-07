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

"""The server-side manager: applies the results of catalog read jobs and answers queries."""

import os
import queue
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME, JobMetaKey, RunStatus, get_job_meta_study
from nvflare.apis.storage import WORKSPACE
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CONTEXT_FILE,
    Selection,
    job_key,
    parse_manifest,
)
from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.server.artifacts import CatalogArtifacts, read_archived_manifest, redact_archive
from nvflare.app_common.fdn.catalog.server.commands import CatalogCommandModule
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically
from nvflare.app_common.fdn.catalog.server.reconcile import Outcome, Reconciler
from nvflare.app_common.fdn.catalog.server.search import SearchPage, SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore, manifest_key
from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules

_UNFINISHED = [RunStatus.SUBMITTED, RunStatus.APPROVED, RunStatus.DISPATCHED, RunStatus.RUNNING]
_LAUNCHED = (RunStatus.DISPATCHED, RunStatus.RUNNING)
_RECOVERY_INTERVAL_S = 30.0


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CatalogManager(FLComponent):
    def __init__(
        self,
        stale_after_s: int = 172800,
        read_limits: dict | None = None,
        register_commands: bool = True,
    ):
        """Runs one catalog read job per Study at a time and applies its result, however the job was submitted: by
        the recipe, `nvflare catalog read` or a PeriodicJobSubmitter. A site is reported stale when its last
        successful read is older than stale_after_s."""
        super().__init__()
        self._stale_after_s = stale_after_s
        self._read_limits = {**DEFAULT_LIMITS, **(read_limits or {})}
        self._store: CatalogStore | None = None
        self._workspace = None
        self._key: bytes | None = None  # the master key; a job gets only the key derived for it
        self._reconciler: Reconciler | None = None
        self._engine = None
        self._artifacts: CatalogArtifacts | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._finished: queue.Queue = queue.Queue()  # ids of finished jobs, handled off the job runner's thread
        self._lock = threading.Lock()
        self._result_lock = threading.Lock()  # serialize ingestion with legacy archive redaction on downloads
        # study -> the one catalog job that holds its read: launched with a context, its result not yet applied
        self._in_flight: dict[str, str] = {}
        self._now = lambda: datetime.now(timezone.utc)
        if register_commands:
            self._register_commands()

    def _register_commands(self):
        """Components are built before the admin server, which reads this list, so the commands are in time."""
        modules = ServerCommandModules.cmd_modules
        modules[:] = [m for m in modules if not isinstance(m, CatalogCommandModule)]
        ServerCommandModules.register_cmd_module(CatalogCommandModule(self))

    # ---- lifecycle -----------------------------------------------------------------------------------------

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            self._start(fl_ctx)
        elif event_type == EventType.SYSTEM_END:
            self._stop.set()
        elif event_type == EventType.SUBMIT_JOB:
            self._refuse_overlap(fl_ctx)
        elif event_type == EventType.BEFORE_JOB_LAUNCH:
            self._hand_over(fl_ctx)
        elif event_type == EventType.BEFORE_JOB_WORKSPACE_ARCHIVE:
            self._take_back(fl_ctx)
        elif event_type == EventType.BEFORE_JOB_DOWNLOAD:
            self._prepare_download(fl_ctx)
        elif event_type in (EventType.JOB_COMPLETED, EventType.JOB_ABORTED):
            # Hand off: this event arrives on the job runner's completion thread, which serves every job.
            self._finished.put(fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID))

    @staticmethod
    def _catalog_study(meta) -> str | None:
        return get_job_meta_study(meta) if isinstance(meta, dict) and meta.get(CATALOG_JOB_META) else None

    def _unfinished_catalog_jobs(self, fl_ctx: FLContext) -> list:
        jobs = self._engine.get_component(SystemComponents.JOB_MANAGER).get_jobs_by_status(_UNFINISHED, fl_ctx)
        return sorted(
            (j for j in jobs if self._catalog_study(j.meta) is not None),
            key=lambda j: j.meta.get(JobMetaKey.SUBMIT_TIME.value, 0.0),
        )

    def _refuse_overlap(self, fl_ctx: FLContext):
        """Refuse a catalog job while another for the same Study is unfinished or not yet applied. A convenience:
        the launch check in _hand_over is what guarantees one read per Study, whatever path created the job."""
        study = self._catalog_study(fl_ctx.get_prop(FLContextKey.JOB_META))
        if self._store is None or study is None:
            return
        with self._lock:
            busy = self._in_flight.get(study)
        if busy is None:
            busy = next(
                (j.job_id for j in self._unfinished_catalog_jobs(fl_ctx) if get_job_meta_study(j.meta) == study), None
            )
        if busy is not None:
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON,
                f"a catalog read for study '{study}' is still running or being applied: job {busy}",
                private=True,
                sticky=False,
            )

    def _hand_over(self, fl_ctx: FLContext):
        """Before a catalog job starts, put what its controller needs in the job's run directory: the job process may
        run in a container that sees only its own job directory, never the server's catalog store. Only one job per
        Study gets it; another one runs without a key, so its result is never applied."""
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        study = self._catalog_study(meta)
        if self._store is None or study is None:
            return
        job_id = meta.get(JobMetaKey.JOB_ID.value)
        with self._lock:
            holder = self._in_flight.setdefault(study, job_id)
        if holder != job_id:
            self.log_warning(
                fl_ctx, f"study '{study}' already has catalog read {holder}; the result of job {job_id} is not applied"
            )
            return
        try:
            run_dir = self._workspace.get_run_dir(job_id)
            context = {
                "key": job_key(self._key, job_id).hex(),
                "previous": self._store.indexed(get_job_meta_study(meta)),
            }
            write_atomically(os.path.join(run_dir, CONTEXT_FILE), context)
        except Exception as e:
            self.log_warning(fl_ctx, f"catalog job context could not be written: {type(e).__name__}")

    def _take_back(self, fl_ctx: FLContext):
        """Capture the result privately, then remove both result and context before archiving for download."""
        with self._result_lock:
            self._capture_result(fl_ctx)

    def _capture_result(self, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        try:
            if self._catalog_study(self._job_meta(job_id, fl_ctx)) is None:
                return
            run_dir = self._workspace.get_run_dir(job_id)
            with self._lock:
                keep = job_id in self._in_flight.values()
            self._artifacts.capture(job_id, run_dir, keep)
        except Exception as e:
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON,
                f"private catalog files could not be secured: {type(e).__name__}",
                private=True,
                sticky=False,
            )

    def _prepare_download(self, fl_ctx: FLContext):
        """Scrub publications from older catalog archives before any download is prepared."""
        with self._result_lock:
            self._redact_download(fl_ctx)

    def _redact_download(self, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        try:
            if self._catalog_study(self._job_meta(job_id, fl_ctx)) is None:
                return
            job_manager = self._engine.get_component(SystemComponents.JOB_MANAGER)
            if WORKSPACE not in job_manager.list_components(job_id, fl_ctx):
                return  # a job that never archived a workspace has nothing to redact
            data = job_manager.get_storage_component(job_id, WORKSPACE, fl_ctx)
            replacement = redact_archive(data)
            if replacement is not None:
                # Preserve a legacy result if completion and download arrive together.
                with self._lock:
                    held = job_id in self._in_flight.values()
                if held:
                    try:
                        self._artifacts.capture_archive(job_id, data, self._key)
                    except (ValueError, TypeError, zipfile.BadZipFile):
                        pass  # invalid results establish no publication; download reveals none
                job_manager.save_workspace(job_id, replacement, fl_ctx)
        except Exception as e:
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON,
                f"catalog archive could not be secured: {type(e).__name__}",
                private=True,
                sticky=False,
            )

    def _start(self, fl_ctx: FLContext):
        self._engine = fl_ctx.get_engine()
        self._workspace = fl_ctx.get_workspace()
        root = self._workspace.get_root_dir()
        self._key = manifest_key(root)
        self._store = CatalogStore(root)
        self._artifacts = CatalogArtifacts(root)
        self._reconciler = Reconciler(self._store, ReadLimits(**self._read_limits))
        try:  # a job that survived a restart still holds its Study's read
            for job in self._unfinished_catalog_jobs(fl_ctx):
                if job.meta.get(JobMetaKey.STATUS.value) in _LAUNCHED:
                    self._in_flight.setdefault(get_job_meta_study(job.meta), job.job_id)
        except Exception as e:
            self.log_warning(fl_ctx, f"unfinished catalog jobs could not be listed: {type(e).__name__}")
        self._thread = threading.Thread(target=self._run, name="catalog-manager", daemon=True)
        self._thread.start()

    def _run(self):
        next_recovery = time.monotonic() + _RECOVERY_INTERVAL_S
        while not self._stop.is_set():
            try:
                job_id = self._finished.get(timeout=1.0)
            except queue.Empty:
                pass
            else:
                try:
                    self._ingest(job_id)
                except Exception as e:
                    self.logger.warning(f"catalog result could not be applied: {type(e).__name__}")
            now = time.monotonic()
            if now >= next_recovery:
                self._recover_finished()
                next_recovery = now + _RECOVERY_INTERVAL_S

    def _recover_finished(self):
        """Some terminal transitions (including deletion) emit no completion event."""
        with self._lock:
            holders = list(self._in_flight.values())
        for job_id in holders:
            try:
                with self._engine.new_context() as fl_ctx:
                    meta = self._job_meta(job_id, fl_ctx)
                if meta is None or meta.get(JobMetaKey.STATUS.value) not in _UNFINISHED:
                    self._ingest(job_id)
            except Exception as e:
                self.logger.warning(f"catalog job recovery failed: {type(e).__name__}")

    # ---- results -------------------------------------------------------------------------------------------

    def _job_meta(self, job_id: str, fl_ctx: FLContext) -> dict | None:
        job = self._engine.get_component(SystemComponents.JOB_MANAGER).get_job(job_id, fl_ctx)
        return job.meta if job is not None else None

    def _ingest(self, job_id: str):
        """Apply the result of the job that holds its Study's read, once, then free the Study for the next read. Any
        other job (not a catalog job, refused at launch, or already applied) changes nothing."""
        with self._result_lock:
            self._apply_job(job_id)

    def _apply_job(self, job_id: str):
        with self._lock:
            study = next((s for s, held in self._in_flight.items() if held == job_id), None)
        if study is None:
            return
        try:
            job_manager = self._engine.get_component(SystemComponents.JOB_MANAGER)
            with self._engine.new_context() as fl_ctx:
                meta = self._job_meta(job_id, fl_ctx)
                if self._catalog_study(meta) != study:
                    return
            now = _iso(self._now())
            try:
                try:
                    manifest = self._artifacts.read(job_id, self._key)
                except FileNotFoundError:
                    with self._engine.new_context() as fl_ctx:
                        data = job_manager.get_storage_component(job_id, WORKSPACE, fl_ctx)
                    manifest = self._read_manifest(data, job_id)
                entries = parse_manifest(manifest)
            except Exception as e:
                self.logger.warning(f"catalog manifest unavailable: {type(e).__name__}")
                for client in self._sites(meta, study):
                    self._store.record_no_result(study, client, "NO_RESULT", now)
                return
            for client, entry in entries.items():
                try:
                    outcome = None
                    if entry.rc == "OK" and entry.result is not None:
                        outcome = self._reconciler.apply(study, client, Selection(entry.datasets), entry.result, now)
                    if outcome is None or outcome is Outcome.UNSUPPORTED:
                        reason = (
                            "UNSUPPORTED_RESULT"
                            if outcome is Outcome.UNSUPPORTED
                            else "NO_RESULT" if entry.rc == "OK" else entry.rc
                        )
                        self._store.record_no_result(study, client, reason, now)
                except Exception as e:
                    self.logger.warning(f"catalog result for a client could not be applied: {type(e).__name__}")
        finally:
            try:
                self._artifacts.discard(job_id)
            finally:
                with self._lock:
                    if self._in_flight.get(study) == job_id:
                        del self._in_flight[study]

    def _sites(self, meta: dict, study: str) -> list[str]:
        """The sites a job was deployed to; for every site, those the catalog knows in the Study."""
        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP.value) or {}
        sites = {s for targets in deploy_map.values() for s in targets if isinstance(s, str)}
        if ALL_SITES in sites:
            return sorted(self._store.indexed(study))
        return sorted(sites - {SERVER_SITE_NAME})

    def _read_manifest(self, data: bytes, job_id: str) -> dict:
        """The manifest the job's controller wrote in the run dir, if its signature checks out. A job's own files
        (and any manifest-like file it ships) are neither in that place nor signed with the server's key."""
        return read_archived_manifest(data, self._key, job_id)

    # ---- queries -------------------------------------------------------------------------------------------

    def search(self, study: str, query: SearchQuery) -> SearchPage:
        return self._store.search(study, query)

    def health(self, study: str) -> list[dict]:
        now = self._now()
        report = []
        for row in self._store.health(study):
            last = row["last_success_at"]
            stale = last is None or now - datetime.fromisoformat(last.replace("Z", "+00:00")) > timedelta(
                seconds=self._stale_after_s
            )
            report.append({**row, "stale": stale})
        return report
