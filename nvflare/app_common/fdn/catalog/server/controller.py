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

"""The server side of a catalog read job: one catalog_read task per site, then catalog_results.json.

The file is written in the job's server workspace, so the job runner saves it to the job store with the job and the
submitter downloads it with the job result. The server keeps no other copy.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.job_def import get_job_meta_study
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    RESULTS_FILE,
    TASK_NAME,
    PayloadError,
    ReasonCode,
    ResultStatus,
    SiteEntry,
    SiteStatus,
    build_catalog_results,
    build_task_payload,
    parse_result,
    task_time_budget,
)
from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.fuel.utils.log_utils import log_progress

_POLL_SECONDS = 0.5
_NO_RESULT = SiteEntry(SiteStatus.NO_RESULT)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CatalogController(Controller):
    def __init__(
        self,
        datasets: dict[str, list[str]] | None = None,
        task_timeout_s: int = 300,
        sites: list[str] | None = None,
    ):
        """datasets maps a site to the names to read there; a site reads the named datasets it shares and skips the
        rest. A site that is not in the mapping, or None, reads everything it shares. The job's clients are the sites
        it was deployed to. sites lists the sites the caller asked for, if it named them: a named site that is not
        among the job's clients (offline, or not scheduled) is recorded as NO_RESULT instead of being left out."""
        super().__init__()
        if datasets is not None and not (
            isinstance(datasets, dict)
            and datasets
            and all(
                isinstance(site, str)
                and site
                and isinstance(names, list)
                and names
                and all(isinstance(n, str) and n for n in names)
                for site, names in datasets.items()
            )
        ):
            raise PayloadError("datasets must map each site to a non-empty list of names")
        if type(task_timeout_s) is not int or task_timeout_s <= 0:
            raise PayloadError("task_timeout_s must be a positive integer")
        if sites is not None and not (
            isinstance(sites, list) and sites and all(isinstance(site, str) and site for site in sites)
        ):
            raise PayloadError("sites must be a non-empty list of site names")
        self.datasets = datasets
        self.sites = sites
        self.task_timeout_s = task_timeout_s
        self.limits = ReadLimits(**DEFAULT_LIMITS)  # the server's own validation limits; sites cannot raise them
        self._results_path: str | None = None
        self._job_id: str | None = None
        self._study: str | None = None
        self._clients: list[str] = []
        self._entries: dict[str, SiteEntry] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()

    def start_controller(self, fl_ctx: FLContext):
        self._results_path = None
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(meta, dict) or meta.get(CATALOG_JOB_META) is not True:
            raise RuntimeError("CatalogController runs only in a catalog read job")
        self._job_id = fl_ctx.get_job_id()
        self._study = get_job_meta_study(meta)
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(self._job_id)
        self._results_path = os.path.join(run_dir, RESULTS_FILE)
        try:  # a file shipped with the job is never a result
            os.remove(self._results_path)
        except FileNotFoundError:
            pass
        self._clients = sorted(c.name for c in self._engine.get_clients())
        missing = sorted(set(self.sites or ()) - set(self._clients))
        if missing:
            self.log_warning(fl_ctx, f"requested sites are not in this job and are recorded as NO_RESULT: {missing}")
        unknown = sorted(set(self.datasets or {}) - set(self._clients) - set(self.sites or ()))
        if unknown:
            self.log_warning(fl_ctx, f"datasets name sites that are not in this job and are ignored: {unknown}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        log_progress(self.logger, f"\n  Federated catalog · {len(self._clients)} sites\n\n  Reading site metadata…")
        # One task per site, because each site may have its own dataset selection.
        done = {name: threading.Event() for name in self._clients}
        for name in self._clients:
            data = Shareable()
            data[CATALOG_READ_KEY] = build_task_payload(self._wanted(name))
            task = Task(
                name=TASK_NAME,
                data=data,
                timeout=self.task_timeout_s,
                before_task_sent_cb=self._set_time_budget,
                result_received_cb=self._record,
                task_done_cb=self._done_callback(done[name]),
            )
            # Use send(), not broadcast(). Each task has exactly one target, so send() delivers it to that site
            # only, and the tasks still run in parallel: send() schedules without waiting, like broadcast().
            #
            # The difference is which data reaches the site. The server runs before_task_sent_cb on every
            # delivery, including a resend, which happens when a site asks for work while its task is still
            # out with no result (for example after a restart or a dropped connection). _set_time_budget
            # writes the time left at that moment into task.data, so the budget shrinks on each delivery.
            #
            # - send() delivers task.data as it stands after the callback, so a resend carries the
            #   reduced budget.
            # - broadcast() takes one snapshot of task.data after the first delivery's callback and
            #   reuses it for every later delivery, resends included (see WFCommServer.broadcast). Every
            #   recipient then gets identical data, which is what a global model needs. But a resend
            #   would carry the first, larger budget: a 60 s task resent after 20 s would still tell the
            #   site 48 s instead of 28 s. The site would read past the task deadline, and the server
            #   would drop the catalogs it completed.
            #
            # Rule: use broadcast() when the payload is fixed at scheduling. Use send() when
            # before_task_sent_cb computes the payload at delivery time, per site or per moment.
            self.send(task, fl_ctx, targets=[name])
        cancelled = False
        for event in done.values():
            while not event.wait(_POLL_SECONDS):
                if abort_signal.triggered:
                    if not cancelled:  # one call cancels every task
                        self.cancel_all_tasks(fl_ctx=fl_ctx)
                        cancelled = True
                    break
        self._write_results()
        if abort_signal.triggered:
            log_progress(self.logger, "  Catalog read cancelled")
        else:
            with self._lock:
                responded = sum(1 for e in self._entries.values() if e.status != SiteStatus.NO_RESULT)
            log_progress(self.logger, f"\n  Catalog read completed · {responded}/{len(self._clients)} sites responded")

    def _wanted(self, client: str) -> list[str] | None:
        return (self.datasets or {}).get(client)

    @staticmethod
    def _set_time_budget(client_task: ClientTask, fl_ctx: FLContext):
        """Give the site the reading time its task has left, measured when the task is sent: the task timeout runs
        from scheduling, and a site may pick its task up late. This runs again on a resend, with less time left.
        Each site has its own task, sent with send(), so every delivery carries the value set here; see
        control_flow."""
        task = client_task.task
        elapsed = time.time() - (task.schedule_time or task.create_time)
        task.data[CATALOG_READ_KEY]["time_budget_s"] = round(task_time_budget(task.timeout, elapsed), 3)

    @staticmethod
    def _done_callback(event: threading.Event):
        # The framework calls the callback with keyword arguments, so the parameter names matter.
        def task_done(task: Task, fl_ctx: FLContext):
            event.set()

        return task_done

    def _record(self, client_task: ClientTask, fl_ctx: FLContext):
        # The site name is the framework's authenticated identity, never a field of the payload.
        name = client_task.client.name
        result = client_task.result
        rc = result.get_return_code() if isinstance(result, Shareable) else None
        payload = result.get(CATALOG_READ_KEY) if rc == ReturnCode.OK else None
        if isinstance(payload, dict):
            wanted = self._wanted(name)
            parsed = parse_result(payload, None if wanted is None else tuple(wanted), self.limits)
            if parsed.unsupported:
                entry = _NO_RESULT
            elif parsed.status == ResultStatus.OK:
                entry = SiteEntry(SiteStatus.OK, None, parsed.catalogs, parsed.unavailable, parsed.skipped)
            else:
                entry = SiteEntry(SiteStatus.FAILED, parsed.reason_code or ReasonCode.INTERNAL)
        else:
            entry = _NO_RESULT
        with self._lock:
            self._entries[name] = entry
        client_task.result = None
        log_progress(self.logger, f"  {name}: {entry.status.value}")

    def _write_results(self):
        if self._results_path is None or self._job_id is None:
            return
        with self._write_lock:  # control_flow and stop_controller can both write
            with self._lock:
                # a requested site that never joined the job is reported, not left out
                names = sorted(set(self._clients) | set(self.sites or ()))
                entries = {name: self._entries.get(name, _NO_RESULT) for name in names}
            document = build_catalog_results(self._study, self._job_id, _now(), entries)
            temporary = f"{self._results_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, allow_nan=False)
            os.replace(temporary, self._results_path)

    def stop_controller(self, fl_ctx: FLContext):
        if self._results_path is not None and not os.path.exists(self._results_path):
            self._write_results()
