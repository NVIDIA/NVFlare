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

"""The server-side workflow of a catalog read job: one catalog.read task per client, then a manifest."""

import json
import os
import threading

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    CONTEXT_FILE,
    MANIFEST_FILE,
    TASK_NAME,
    ManifestEntry,
    PayloadError,
    ReasonCode,
    ResultStatus,
    Selection,
    build_manifest,
    build_result,
    build_task_payload,
    manifest_mac,
    parse_result,
)
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically
from nvflare.fuel.utils.log_utils import log_progress

_POLL_SECONDS = 0.5


class CatalogController(Controller):
    def __init__(self, datasets: dict[str, list[str]] | None = None, task_timeout_s: int = 300):
        """datasets maps a site to the names to read there; a site reads the named datasets it has and skips the
        rest. A site that is not in the mapping, or None, reads everything its policy shares. The job's clients are
        the sites it was deployed to."""
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
        self.datasets = datasets
        self.task_timeout_s = task_timeout_s
        self._previous_names: dict[str, list[str]] = {}
        self._manifest_path: str | None = None
        self._job_id: str | None = None
        self._key: bytes | None = None
        self._clients: list[str] = []
        self._entries: dict[str, ManifestEntry] = {}
        self._lock = threading.Lock()

    def start_controller(self, fl_ctx: FLContext):
        self._key = None
        self._manifest_path = None
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(meta, dict) or meta.get(CATALOG_JOB_META) is not True:
            raise RuntimeError("catalog collection requires an admitted catalog job")
        workspace = fl_ctx.get_engine().get_workspace()
        self._job_id = fl_ctx.get_job_id()
        run_dir = workspace.get_run_dir(self._job_id)
        self._load_context(os.path.join(run_dir, CONTEXT_FILE), fl_ctx)
        if self._key is None:
            raise RuntimeError("catalog collection requires the manager-issued signing context")
        self._manifest_path = os.path.join(run_dir, MANIFEST_FILE)
        self._clients = sorted(c.name for c in self._engine.get_clients())
        unknown = sorted(set(self.datasets or {}) - set(self._clients))
        if unknown:
            self.log_warning(fl_ctx, f"datasets name sites that are not in this job and are ignored: {unknown}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        if self._key is None or self._manifest_path is None:
            raise RuntimeError("catalog collection has not been admitted")
        log_progress(
            self.logger, f"\n  Federated catalog · {len(self._clients)} sites\n\n  Reading site-approved metadata…"
        )
        # One task per client because the names the server already holds differ by client.
        done = {name: threading.Event() for name in self._clients}
        for name in self._clients:
            data = Shareable()
            data[CATALOG_READ_KEY] = build_task_payload(self._wanted(name), self._previous(name))
            task = Task(
                name=TASK_NAME,
                data=data,
                timeout=self.task_timeout_s,
                result_received_cb=self._record,
                task_done_cb=self._done_callback(done[name]),
            )
            self.broadcast(task, fl_ctx, targets=[name], min_responses=1)
        cancelled = False
        for event in done.values():
            while not event.wait(_POLL_SECONDS):
                if abort_signal.triggered:
                    if not cancelled:  # one call cancels every task
                        self.cancel_all_tasks(fl_ctx=fl_ctx)
                        cancelled = True
                    break
        self._write_manifest()
        if abort_signal.triggered:
            log_progress(self.logger, "  Catalog collection cancelled")
        else:
            with self._lock:
                responded = len(set(self._entries) & set(self._clients))
            log_progress(
                self.logger, f"\n  Catalog collection completed · {responded}/{len(self._clients)} sites responded"
            )

    def _load_context(self, path: str, fl_ctx: FLContext):
        """What the server hands over in the job's own run directory, which every launcher exposes to the job: the
        key of this job and the dataset names the server holds. Remove this copy immediately; JobRunner also
        removes the parent copy before archiving when a container has its own workspace."""
        self._key = None
        self._previous_names = {}
        try:
            with open(path, encoding="utf-8") as f:
                context = json.load(f)
            key = bytes.fromhex(context["key"])
            if len(key) != 32:
                raise ValueError("invalid catalog signing key")
            previous = {c: list(names) for c, names in context["previous"].items()}
            os.remove(path)  # only a file that was understood is removed; the server removes any other
            self._key, self._previous_names = key, previous
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self.log_warning(fl_ctx, "no valid catalog context from the server; collection is refused")

    def _wanted(self, client: str) -> list[str] | None:
        return (self.datasets or {}).get(client)

    def _previous(self, client: str) -> list[str]:
        names = self._previous_names.get(client, [])
        wanted = self._wanted(client)
        return names if wanted is None else [n for n in names if n in wanted]

    @staticmethod
    def _done_callback(event: threading.Event):
        # The framework calls the callback with keyword arguments, so the parameter names matter.
        def task_done(task: Task, fl_ctx: FLContext):
            event.set()

        return task_done

    def _record(self, client_task: ClientTask, fl_ctx: FLContext):
        # The client name is the framework's authenticated identity, never a field of the payload.
        result = client_task.result
        rc = result.get_return_code() if isinstance(result, Shareable) else ReturnCode.EXECUTION_EXCEPTION
        payload = result.get(CATALOG_READ_KEY) if rc == ReturnCode.OK else None
        name = client_task.client.name
        wanted = self._wanted(name)
        if isinstance(payload, dict):
            try:
                # A site result must be safe for the shared JSON manifest. One bad site must not
                # prevent other sites' publications or policy restrictions from being applied.
                # It must also come back unchanged from JSON: integer keys would return as strings and
                # re-sort, which would invalidate the signature for every site in the job.
                if json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)) != payload:
                    raise ValueError("site result does not survive a JSON round trip")
            except (TypeError, ValueError, OverflowError, RecursionError):
                parsed = parse_result(
                    {**payload, "status": ResultStatus.FAILED.value},
                    Selection(None if wanted is None else tuple(wanted)),
                )
                payload = build_result(
                    ResultStatus.FAILED,
                    withheld=parsed.withheld,
                    unavailable=parsed.unavailable,
                    reason_code=ReasonCode.INTERNAL,
                )
        with self._lock:
            self._entries[name] = ManifestEntry(
                rc, payload if isinstance(payload, dict) else None, None if wanted is None else tuple(wanted)
            )
        client_task.result = None
        message = "catalog response received" if rc == ReturnCode.OK else "catalog request failed"
        log_progress(self.logger, f"  {name}: {message}")

    def _write_manifest(self):
        if self._key is None or self._manifest_path is None or self._job_id is None:
            return  # never emit unsigned publications, including during failed-start cleanup
        with self._lock:
            entries = {name: self._entries.get(name, ManifestEntry(ReturnCode.TIMEOUT, None)) for name in self._clients}
        manifest = build_manifest(entries)
        mac = manifest_mac(self._key, self._job_id, manifest)
        write_atomically(self._manifest_path, {"manifest": manifest, "mac": mac})

    def stop_controller(self, fl_ctx: FLContext):
        if self._manifest_path is not None:
            self._write_manifest()
