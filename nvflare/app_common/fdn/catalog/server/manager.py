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

import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SiteType, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus, get_job_meta_study
from nvflare.apis.storage import WORKSPACE
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CONTEXT_FILE,
    Selection,
    job_key,
    parse_manifest,
)
from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.server.artifacts import CatalogArtifacts, read_archived_manifest, redact_download
from nvflare.app_common.fdn.catalog.server.commands import CatalogCommandModule, NotReady
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically
from nvflare.app_common.fdn.catalog.server.reconcile import Outcome, Reconciler
from nvflare.app_common.fdn.catalog.server.search import SearchPage, SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore, manifest_key
from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules
from nvflare.utils.job_launcher_utils import get_job_launcher_spec

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
        """Applies catalog results as they arrive: the last applied update for each dataset wins.
        Reads may overlap, including within a Study. A site is reported stale when its last successful read
        is older than stale_after_s."""
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
        self._result_lock = threading.Lock()  # serialize parent inbox capture with ingestion
        self._pending: set[str] = set()  # launched jobs whose results still need processing; never blocks a read
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
        return get_job_meta_study(meta) if isinstance(meta, dict) and meta.get(CATALOG_JOB_META) is True else None

    def _validate_catalog_job(self, job_id: str, meta: dict, server_name: str = SiteType.SERVER):
        """Only the built-in collector may receive a key; the catalog marker is not code authorization."""
        # Resolve exactly as the launchers do: defaults, site overrides and legacy resource_spec.
        # This gate is independent of the locally derived BYOC flag.
        for name in {SiteType.SERVER, server_name}:
            if any(get_job_launcher_spec(meta, name, mode) for mode in ("docker", "k8s", "slurm")):
                raise ValueError("catalog server execution settings must be site-owned, not job-supplied")
        custom = Path(self._workspace.get_app_custom_dir(job_id))
        if meta.get(AppValidationKey.BYOC) or (custom.is_dir() and any(custom.iterdir())):
            raise ValueError("catalog jobs cannot contain server-side BYOC code or custom entrypoints")
        path = self._workspace.get_server_app_config_file_path(job_id)
        config = json.loads(Path(path).read_text())
        allowed = {"format_version", "workflows", "components", "task_data_filters", "task_result_filters"}
        if not isinstance(config, dict) or set(config) - allowed or config.get("format_version") != 2:
            raise ValueError("catalog jobs require the standard server JSON configuration")
        workflows = config.get("workflows")
        if not isinstance(workflows, list) or len(workflows) != 1:
            raise ValueError("catalog jobs require exactly one CatalogController")
        workflow = workflows[0]
        if (
            not isinstance(workflow, dict)
            or set(workflow) - {"id", "path", "args"}
            or not isinstance(workflow.get("id", "controller"), str)
            or workflow.get("path") != "nvflare.app_common.fdn.catalog.server.controller.CatalogController"
        ):
            raise ValueError("catalog jobs require the built-in CatalogController")
        args = workflow.get("args", {})
        if not isinstance(args, dict) or set(args) - {"datasets", "task_timeout_s"}:
            raise ValueError("unsupported CatalogController arguments")
        CatalogController(**args)  # validate only the built-in controller's data arguments
        components = config.get("components", [])
        if not isinstance(components, list) or any(
            not isinstance(component, dict)
            or set(component) - {"id", "path", "args"}
            or not isinstance(component.get("id"), str)
            or component.get("path") != "nvflare.app_common.logging.job_log_receiver.JobLogReceiver"
            or component.get("args", {}) != {}
            for component in components
        ):
            raise ValueError("catalog jobs permit only the standard log receiver alongside the controller")
        if config.get("task_data_filters") or config.get("task_result_filters"):
            raise ValueError("catalog jobs cannot install server task filters")

    def _hand_over(self, fl_ctx: FLContext):
        """Give each catalog job its own signing key and a snapshot of indexed names, in its run directory.
        The job process may run in a container that cannot see the server's catalog store."""
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        study = self._catalog_study(meta)
        if self._store is None or study is None:
            return
        job_id = meta.get(JobMetaKey.JOB_ID.value)
        try:
            self._validate_catalog_job(job_id, meta, fl_ctx.get_identity_name() or SiteType.SERVER)
        except Exception as e:
            self.log_warning(fl_ctx, f"catalog job configuration rejected: {e}")
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON, "catalog job configuration is not permitted", private=True, sticky=False
            )
            return
        try:
            run_dir = self._workspace.get_run_dir(job_id)
            context = {
                "key": job_key(self._key, job_id).hex(),
                "previous": self._store.indexed(get_job_meta_study(meta)),
            }
            write_atomically(os.path.join(run_dir, CONTEXT_FILE), context)
            with self._lock:
                self._pending.add(job_id)
        except Exception as e:
            self.log_warning(fl_ctx, f"catalog job context could not be written: {type(e).__name__}")
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON, "catalog job context could not be written", private=True, sticky=False
            )

    def _take_back(self, fl_ctx: FLContext):
        """Capture the result privately, then remove both result and context before archiving for download."""
        with self._result_lock:
            self._capture_result(fl_ctx)

    def _capture_result(self, fl_ctx: FLContext):
        if self._artifacts is None:  # the service did not start, so it handed nothing over; do not block other jobs
            return
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        try:
            run_dir = self._workspace.get_run_dir(job_id)
            with self._lock:
                keep = job_id in self._pending
            self._artifacts.capture(job_id, run_dir, keep)
        except Exception as e:
            fl_ctx.set_prop(
                FLContextKey.JOB_BLOCK_REASON,
                f"private catalog files could not be secured: {type(e).__name__}",
                private=True,
                sticky=False,
            )

    def _prepare_download(self, fl_ctx: FLContext):
        """Reserved catalog files never leave in a staged workspace, regardless of the job's marker."""
        staged = fl_ctx.get_prop(FLContextKey.JOB_DOWNLOAD_FILES) or {}
        path = staged.get(WORKSPACE)
        if path is None or not os.path.exists(path):
            return  # no workspace archive was stored (an aborted job, or its archive failed): nothing to scrub
        try:
            redact_download(path)
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
        try:  # continue tracking jobs that were running when the server restarted
            jobs = self._engine.get_component(SystemComponents.JOB_MANAGER).get_jobs_by_status(list(_LAUNCHED), fl_ctx)
            self._pending.update(job.job_id for job in jobs if self._catalog_study(job.meta) is not None)
        except Exception as e:
            self.log_warning(fl_ctx, f"running catalog jobs could not be listed: {type(e).__name__}")
        try:
            self._pending.update(self._artifacts.pending_jobs())
        except OSError as e:
            self.log_warning(fl_ctx, f"retained catalog results could not be listed: {type(e).__name__}")
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
        """Recover missed terminal events and retained inbox results, including after restart."""
        try:
            retained = self._artifacts.pending_jobs()
            with self._lock:
                self._pending.update(retained)
        except OSError as e:
            self.logger.warning(f"retained catalog results could not be listed: {type(e).__name__}")
        with self._lock:
            pending = list(self._pending)
        for job_id in pending:
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
        """Apply a pending job's result once; retry storage failures without blocking other catalog reads."""
        with self._result_lock:
            self._apply_job(job_id)

    def _apply_job(self, job_id: str):
        with self._lock:
            if job_id not in self._pending:
                return
        retry = False
        try:
            job_manager = self._engine.get_component(SystemComponents.JOB_MANAGER)
            with self._engine.new_context() as fl_ctx:
                meta = self._job_meta(job_id, fl_ctx)
                study = self._catalog_study(meta)
                if study is None:
                    return
            now = _iso(self._now())
            try:
                try:
                    manifest = self._artifacts.read(job_id, self._key)
                except FileNotFoundError:
                    with self._engine.new_context() as fl_ctx:
                        data = job_manager.get_storage_component(job_id, WORKSPACE, fl_ctx)
                    manifest = self._read_manifest(data, job_id)
                    self._artifacts.capture_archive(job_id, data, self._key)
                entries = parse_manifest(manifest)
            except OSError as e:
                if not isinstance(e, FileNotFoundError):
                    raise
                self.logger.warning(f"catalog job {job_id} left no result: {type(e).__name__}")
                return
            except Exception as e:
                # No signed manifest says the job failed, not which site did: no site's health changes.
                self.logger.warning(f"catalog job {job_id} left no usable result: {type(e).__name__}")
                return
            outstanding = self._artifacts.pending_jobs()
            for client, entry in entries.items():
                try:
                    outcome = None
                    if entry.rc == "OK" and entry.result is not None:
                        outcome = self._reconciler.apply(
                            study,
                            client,
                            Selection(entry.datasets),
                            entry.result,
                            now,
                            application_id=job_id,
                            pending_applications=outstanding,
                        )
                    if outcome is None or outcome is Outcome.UNSUPPORTED:
                        reason = (
                            "UNSUPPORTED_RESULT"
                            if outcome is Outcome.UNSUPPORTED
                            else "NO_RESULT" if entry.rc == "OK" else entry.rc
                        )
                        self._store.record_no_result(
                            study, client, reason, now, application_id=job_id, pending_applications=outstanding
                        )
                except OSError as e:
                    retry = True
                    self.logger.warning(f"catalog result for a client will be retried: {type(e).__name__}")
                except Exception as e:
                    self.logger.warning(f"catalog result for a client could not be applied: {type(e).__name__}")
        except Exception as e:
            retry = True
            self.logger.warning(f"catalog result application will be retried: {type(e).__name__}")
        finally:
            cleanups = (self._remove_context,) if retry else (self._artifacts.discard, self._remove_context)
            for cleanup in cleanups:
                try:
                    cleanup(job_id)
                except Exception as e:
                    retry = True
                    self.logger.warning(f"catalog job cleanup failed: {type(e).__name__}")
            if not retry:
                with self._lock:
                    self._pending.discard(job_id)

    def _context_path(self, job_id: str) -> Path | None:
        try:
            return Path(self._workspace.get_run_dir(job_id)) / CONTEXT_FILE
        except Exception:  # applying the result does not depend on finding the run directory
            return None

    def _remove_context(self, job_id: str):
        """The hand-over file holds the job's key; a job that never started leaves it behind."""
        path = self._context_path(job_id)
        try:
            if path is not None:
                path.unlink(missing_ok=True)
        except OSError as e:
            self.logger.warning(f"catalog job context could not be removed: {type(e).__name__}")

    def _read_manifest(self, data: bytes, job_id: str) -> dict:
        """The manifest the job's controller wrote in the run dir, if its signature checks out. A job's own files
        (and any manifest-like file it ships) are neither in that place nor signed with the server's key."""
        return read_archived_manifest(data, self._key, job_id)

    # ---- queries -------------------------------------------------------------------------------------------

    def _ready_store(self) -> CatalogStore:
        if self._store is None:
            raise NotReady("the catalog service is still starting")
        return self._store

    def search(self, study: str, query: SearchQuery) -> SearchPage:
        return self._ready_store().search(study, query)

    def health(self, study: str) -> list[dict]:
        now = self._now()
        report = []
        for row in self._ready_store().health(study):
            last = row["last_success_at"]
            stale = last is None or now - datetime.fromisoformat(last.replace("Z", "+00:00")) > timedelta(
                seconds=self._stale_after_s
            )
            report.append({**row, "stale": stale})
        return report
