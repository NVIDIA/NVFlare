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

"""Online CollabAPI server workflow shared by FedAvg, FedBuff, and FedRevive."""

import gc
import heapq
import json
import os
import queue
import random
import shutil
import tempfile
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from data import load_manifest, test_transform
from dfkd import ClassProportionProxyEstimator, DFKDConfig, DFKDReviver
from fedrevive import (
    FIGURE_2_METHOD_CONFIGS,
    BufferedUpdate,
    FedReviveMode,
    InTimeUpdateBuffer,
    Method,
    TeacherBuffer,
    cosine_staleness_beta,
    process_update,
)
from model import create_model, get_model_params, load_model_params
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from nvflare.app_common.utils.tensor_disk_offload_context import cleanup_tensor_disk_offload, setup_tensor_disk_offload
from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict
from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.fuel.utils.log_utils import get_obj_logger


@dataclass
class _DiskSnapshot:
    """One immutable global version shared by all assignments based on it.

    Only this small descriptor stays in scheduler memory.  The tensors live in
    the run workspace and are memory-mapped briefly when Collab must send them
    or when an accepted client model must be differenced against them.
    """

    model_version: int
    path: str
    users: int = 0


class _SnapshotStore:
    """Reference-count assignment snapshots and remove them at last use.

    The arbitrary-arrival scheduler can have K logical assignments based on
    many old global versions.  Those snapshots are correctness state: without
    the exact starting weights, a stale returned model cannot be converted to
    the update that the client actually trained.  They are not hot working
    state, however, so retaining their tensors in ``_ActiveJob`` needlessly
    grows anonymous memory with K and previously caused host reclaim stalls.

    A version is serialized once and shared by every assignment created before
    the next aggregation.  ``torch.load(..., mmap=True)`` keeps loading from
    turning the whole snapshot into anonymous memory; callers hold the mapping
    only for the duration of dispatch or delta computation.  The final user
    deletes the file immediately, bounding disk usage by live assignments.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # A killed run can leave this algorithm-owned cache behind.  It is
        # derived state, never a checkpoint, so a fresh workflow must not reuse
        # it and may safely replace it inside its own run directory.
        shutil.rmtree(self.root_dir, ignore_errors=True)
        os.makedirs(self.root_dir, exist_ok=True)
        self._by_version: dict[int, _DiskSnapshot] = {}

    def retain(self, model_version: int, model: dict[str, torch.Tensor], users: int) -> _DiskSnapshot:
        if users < 1:
            raise ValueError("snapshot users must be positive")
        snapshot = self._by_version.get(model_version)
        if snapshot is None:
            path = os.path.join(self.root_dir, f"model_version_{model_version}.pt")
            staging_path = f"{path}.staging"
            try:
                # Serialization is synchronous, so no clone is needed: the
                # server replaces rather than mutates _global_model when a new
                # version is produced.
                torch.save(model, staging_path)
                os.replace(staging_path, path)
            finally:
                try:
                    os.remove(staging_path)
                except FileNotFoundError:
                    pass
            snapshot = _DiskSnapshot(model_version=model_version, path=path)
            self._by_version[model_version] = snapshot
        snapshot.users += users
        return snapshot

    def load(self, snapshot: _DiskSnapshot) -> dict[str, torch.Tensor]:
        if self._by_version.get(snapshot.model_version) is not snapshot or snapshot.users < 1:
            raise RuntimeError(f"snapshot for model version {snapshot.model_version} is no longer active")
        return torch.load(snapshot.path, map_location="cpu", weights_only=True, mmap=True)

    def release(self, snapshot: _DiskSnapshot) -> None:
        if self._by_version.get(snapshot.model_version) is not snapshot or snapshot.users < 1:
            raise RuntimeError(f"invalid release for model version {snapshot.model_version}")
        snapshot.users -= 1
        if snapshot.users == 0:
            del self._by_version[snapshot.model_version]
            try:
                os.remove(snapshot.path)
            except FileNotFoundError:
                pass

    def cleanup(self) -> None:
        self._by_version.clear()
        shutil.rmtree(self.root_dir, ignore_errors=True)


@dataclass
class _ActiveJob:
    """Logical assignment state that must survive until its upload is accepted.

    A physical Collab site is only a bounded execution worker.  The logical
    client, assignment id, starting model version, and exact starting weights
    define the simulated job and therefore its eventual staleness.  Keeping
    this state server-side lets physical workers be reused without changing the
    paper's logical K-client schedule.
    """

    logical_name: str
    assignment_id: int
    model_version: int
    # Shared by every live assignment based on this global version.  The exact
    # tensors are disk-backed because most jobs are pending or waiting for a
    # later simulated event rather than using the snapshot at this instant.
    base_snapshot: _DiskSnapshot
    train_seed: int
    finish_time: float | None = None
    logical_released: bool = False


@dataclass
class _SimEvent:
    """One deterministic download/train/upload transition in simulated time."""

    finish_time: float
    phase: str
    assignment_id: int

    def __lt__(self, other):
        return self.finish_time < other.finish_time


@dataclass
class _ClientOutcome:
    """Physical RPC completion, possibly waiting for its logical upload event."""

    physical_name: str
    assignment_id: int | None = None
    result: object = None
    error: object = None


class FedReviveServer:
    """Run the Figure 2 methods through one online K/B/O scheduler.

    Real Collab calls are allowed to complete in any order.  Separately, the
    seeded event heap defines the paper simulator's logical arrival order from
    the configured per-client delays.  The server only feeds an update to
    FedAvg/FedBuff/FedRevive when its simulated upload event is next.  This is
    important for reproducibility: changing a delay schedule can substantially
    change which stale updates share a FedBuff buffer and hence its curve.
    """

    def __init__(
        self,
        method: Method | str,
        data_root: str,
        prepared_data_root: str,
        max_time: float = 200.0,
        num_active_jobs: int | None = None,
        buffer_size: int | None = None,
        min_open_slots: int | None = None,
        server_lr: float | None = None,
        call_timeout: float = 3600.0,
        max_parallel: int = 12,
        device: str | None = None,
        eval_batch_size: int = 300,
        eval_interval: int | None = None,
        in_time: bool = True,
        setup_seed: int = 10,
        run_seed: int = 10,
        max_model_versions: int = 50000,
        fedrevive_mode: FedReviveMode | str = FedReviveMode.REPRODUCTION,
    ):
        self.method = Method(method)
        self.fedrevive_mode = FedReviveMode(fedrevive_mode)
        if self.method is not Method.FEDREVIVE and self.fedrevive_mode is not FedReviveMode.REPRODUCTION:
            raise ValueError("paper-aligned mode is only valid for FedRevive")
        preset = FIGURE_2_METHOD_CONFIGS[self.method]
        self.data_root = data_root
        self.prepared_data_root = prepared_data_root
        self.max_time = float(max_time)
        self.num_active_jobs = preset.num_active_jobs if num_active_jobs is None else int(num_active_jobs)
        self.buffer_size = preset.buffer_size if buffer_size is None else int(buffer_size)
        self.min_open_slots = preset.min_open_slots if min_open_slots is None else int(min_open_slots)
        self.server_lr = preset.server_lr if server_lr is None else float(server_lr)
        self.call_timeout = float(call_timeout)
        self.max_parallel = int(max_parallel)
        self.device = device
        self.eval_batch_size = int(eval_batch_size)
        self.eval_interval = (
            (1 if self.method is Method.FEDAVG else 10) if eval_interval is None else int(eval_interval)
        )
        self.in_time = bool(in_time)
        self.setup_seed = int(setup_seed)
        self.run_seed = int(run_seed)
        self.max_model_versions = int(max_model_versions)
        if self.max_time <= 0 or self.num_active_jobs < 1 or self.buffer_size < 1:
            raise ValueError("max_time, num_active_jobs, and buffer_size must be positive")
        if not 1 <= self.min_open_slots <= self.num_active_jobs:
            raise ValueError("min_open_slots must be between 1 and num_active_jobs")
        if self.server_lr <= 0 or self.eval_interval < 1 or self.max_model_versions < 1 or self.max_parallel < 0:
            raise ValueError(
                "server_lr, eval_interval, and max_model_versions must be positive; max_parallel must be >= 0"
            )

        self.logger = get_obj_logger(self)
        # There are two completion domains.  _outcomes receives host/RPC
        # completions from callback threads.  _completed_outcomes indexes them
        # by assignment until the event heap says that logical upload is due.
        # Tensor disk offload makes the latter map a collection of lightweight
        # LazyTensorDict handles rather than a collection of full models.
        self._outcomes = queue.Queue()
        self._completed_outcomes: dict[int, _ClientOutcome] = {}

        # Logical scheduling state is intentionally independent from the
        # smaller physical execution pool.  K can therefore remain 100 as in
        # the paper even when only a few local simulator processes are used.
        self._active_jobs: dict[int, _ActiveJob] = {}
        self._physical_jobs: dict[str, int] = {}
        self._pending_assignments = deque()
        self._event_heap = []
        self._available_physical: list[str] = []
        self._available_logical: list[str] = []
        self._num_open_slots = 0
        self._next_assignment_id = 0
        self._model_version = 0
        self._simulated_time = 0.0
        self._global_model = None
        # In-time accumulation bounds FedAvg/FedBuff storage independently of
        # B.  --no-in-time is retained for inspecting individual contributions.
        self._update_buffer: list[BufferedUpdate] | InTimeUpdateBuffer = InTimeUpdateBuffer() if self.in_time else []
        self._manifest = None
        # This dedicated RNG is the reproducibility control for logical-client
        # selection and sampled delay events.  Do not replace it with wall-clock
        # completion timing: FedBuff is sensitive to the resulting arrival and
        # staleness sequence, particularly under the paper's shifted schedule.
        self._runtime_rng = np.random.RandomState(self.run_seed)
        self._dfkd_config = DFKDConfig(
            generation_interval=10 if self.fedrevive_mode is FedReviveMode.PAPER_ALIGNED else 1
        )
        self._teacher_buffer = TeacherBuffer(self._dfkd_config.teacher_buffer_size)
        self._reviver = None
        self._proxy_estimator = None
        self._test_loader = None
        self._eval_model = None
        self._history = []
        self._writer = None
        self._run_dir = None
        self._tensor_disk_offload_context = None
        self._snapshot_store = None
        # Nonblocking Group calls need their ResultQueue consumed so failures
        # reach the scheduler.  A fixed daemon pool bounds that bookkeeping by
        # physical sites instead of creating one long-lived watcher per call.
        self._call_watch_queue = queue.Queue()
        self._call_watchers = []

    def _device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _init_outputs(self):
        self._run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        os.makedirs(self._run_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=os.path.join(self._run_dir, "tensorboard"))

    def _setup_tensor_disk_offload(self):
        """Stage out-of-order model payloads on disk until logically due.

        With arbitrary simulated arrival sequences, a fast physical worker may
        return an update whose upload event is far behind another assignment.
        Retaining every such model in RAM caused long runs to exhaust the host
        and hang.  NVFlare's tensor offload replaces received tensor values
        with lazy references, while scalar metrics and metadata stay resident.
        """

        # Keep the offload tree with the user-selected persistent workspace,
        # not /tmp (which may be a small tmpfs).  Change tempfile configuration
        # only while the NVFlare helper allocates its tree, and restore it even
        # if setup fails.  cleanup_tensor_disk_offload handles every exit path.
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = self._run_dir
        try:
            context = setup_tensor_disk_offload(
                engine=collab.fl_ctx.get_engine(),
                enabled=True,
                job_id=collab.fl_ctx.get_job_id(),
            )
        finally:
            tempfile.tempdir = previous_tempdir
        if not context.applied:
            raise RuntimeError("tensor disk offload requires an active NVFlare Cell")
        self.logger.info(f"completed client results will be staged under {context.root_dir}")
        return context

    @staticmethod
    def _release_outcome_result(outcome: _ClientOutcome):
        """Drop one queued result and unlink any lazy tensor backing files."""

        result = outcome.result
        # Break the owning reference first so an exception cannot leave a full
        # result reachable through the scheduler bookkeeping.
        outcome.result = None
        if isinstance(result, (tuple, list)) and result and isinstance(result[0], LazyTensorDict):
            result[0].cleanup()

    def _clear_completed_outcomes(self):
        for outcome in self._completed_outcomes.values():
            self._release_outcome_result(outcome)
        self._completed_outcomes.clear()

    def _clear_active_jobs(self):
        """Release every assignment's reference to its disk snapshot."""

        active_jobs = list(self._active_jobs.values())
        self._active_jobs.clear()
        if self._snapshot_store:
            for job in active_jobs:
                self._snapshot_store.release(job.base_snapshot)

    def _start_call_watchers(self, count: int):
        """Start a bounded failure-monitor pool for nonblocking Group calls."""

        for index in range(count):
            watcher = threading.Thread(
                target=self._consume_call_results,
                daemon=True,
                name=f"fedrevive_call_watcher_{index}",
            )
            watcher.start()
            self._call_watchers.append(watcher)

    def _consume_call_results(self):
        while True:
            item = self._call_watch_queue.get()
            try:
                if item is None:
                    return
                results, assignment_by_physical = item
                self._watch_call_failures(results, assignment_by_physical)
            finally:
                self._call_watch_queue.task_done()

    def _stop_call_watchers(self):
        for _ in self._call_watchers:
            self._call_watch_queue.put(None)
        # The threads are daemonized because an abort may leave an underlying
        # Collab result iterator waiting.  Completed workers exit promptly;
        # incomplete ones must not hold process shutdown or system resources.
        for watcher in self._call_watchers:
            watcher.join(timeout=0.1)
        self._call_watchers.clear()

    @staticmethod
    def _materialize_result(result):
        """Load only the logically next update from disk into server memory."""

        updated_model, metrics, metadata = result
        updated_model = {
            name: (value.materialize() if callable(getattr(value, "materialize", None)) else value)
            for name, value in updated_model.items()
        }
        return updated_model, metrics, metadata

    def _init_data(self):
        self._manifest = load_manifest(self.prepared_data_root)
        logical_count = int(self._manifest["num_logical_clients"])
        if self.fedrevive_mode is FedReviveMode.PAPER_ALIGNED:
            # The same prepared splits support the oracle reproduction mode,
            # but the published algorithm must make those label histograms
            # unavailable to its server-side update path.
            self._manifest.pop("class_proportions", None)
        self._available_logical = [f"client-{index}" for index in range(logical_count)]
        test_set = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=False,
            transform=test_transform(),
        )
        self._test_loader = DataLoader(
            test_set,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _sample_train_duration(self, logical_name: str) -> float:
        # Runtime profiles are persistent logical-client properties.  Sampling
        # from them here, rather than measuring physical wall time, makes an
        # experiment's arrival sequence reproducible and allows the default and
        # shifted paper schedules to be compared on the same execution pool.
        profile = self._manifest["runtime_profiles"][logical_name]
        return float(self._runtime_rng.exponential(float(profile["train_mean"])))

    def _sample_upload_duration(self, logical_name: str) -> float:
        profile = self._manifest["runtime_profiles"][logical_name]
        upload_mean = float(profile["upload_mean"])
        # The uniform half-width is part of the experimental delay schedule,
        # rather than a host-performance tuning knob.  The fallback preserves
        # manifests prepared before this field was made explicit.
        half_width = float(profile.get("upload_half_width", 0.02))
        return float(self._runtime_rng.uniform(max(0.0, upload_mean - half_width), upload_mean + half_width))

    def _evaluate(self, final: bool = False):
        device = self._device()
        if self._eval_model is None:
            self._eval_model = create_model()
        load_model_params(self._eval_model, self._global_model, target_device=device)
        self._eval_model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in self._test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self._eval_model(inputs)
                loss_sum += torch.nn.functional.cross_entropy(outputs, labels, reduction="sum").item()
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        record = {
            "model_version": self._model_version,
            "simulated_time": self._simulated_time,
            "accuracy": correct / total,
            "test_loss": loss_sum / total,
            "final": final,
        }
        self._history.append(record)
        self._writer.add_scalar("test/accuracy", record["accuracy"], self._simulated_time)
        self._writer.flush()
        with open(os.path.join(self._run_dir, "accuracy_history.json"), "w", encoding="utf-8") as stream:
            json.dump(self._history, stream, indent=2)
        self.logger.info(
            f"[{collab.call_info}] evaluation version={self._model_version} "
            f"time={self._simulated_time:.2f} accuracy={record['accuracy']:.4f}"
        )

    # @collab.main exposes this method as the server workflow entry point.  Its
    # nonblocking calls invoke the client's matching @collab.publish method;
    # callbacks collect physical results while this loop accepts them in seeded
    # simulated completion-time order.
    @collab.main
    def execute(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        random.seed(self.setup_seed)
        np.random.seed(self.setup_seed)
        torch.manual_seed(self.setup_seed)
        self._init_outputs()
        self._snapshot_store = _SnapshotStore(os.path.join(self._run_dir, "assignment_snapshots"))
        self._init_data()
        initial_model = get_model_params(create_model(), target_device="cpu")
        inherited_model = collab.get_prop(ContextKey.RESULT, initial_model)
        self._global_model = {name: value.detach().cpu().clone() for name, value in inherited_model.items()}
        if self.method is Method.FEDREVIVE:
            self._reviver = DFKDReviver(
                self._device(), output_dir=os.path.join(self._run_dir, "dfkd"), config=self._dfkd_config
            )
            if self.fedrevive_mode is FedReviveMode.PAPER_ALIGNED:
                # The estimator consumes only ordinary uploaded weights and
                # runs entirely on the server.  No label histogram or other
                # auxiliary metadata crosses the Collab boundary.
                self._proxy_estimator = ClassProportionProxyEstimator(
                    device=self._device(),
                    num_uploads=self._dfkd_config.proxy_num_uploads,
                    temperature=self._dfkd_config.proxy_temperature,
                    batch_size=self._dfkd_config.proxy_batch_size,
                    seed=self.run_seed,
                )

        physical_names = [client.name for client in collab.clients]
        self._available_physical = physical_names
        self._num_open_slots = self.num_active_jobs
        self._start_call_watchers(len(physical_names))

        self.logger.info(
            f"[{collab.call_info}] method={self.method.value} K={self.num_active_jobs} "
            f"B={self.buffer_size} O={self.min_open_slots} in_time={self.in_time} max_time={self.max_time} "
            f"fedrevive_mode={self.fedrevive_mode.value}"
        )
        try:
            # Offload must be active before dispatch: otherwise every early
            # physical result waiting for a later simulated event holds a full
            # model in server RAM.
            self._tensor_disk_offload_context = self._setup_tensor_disk_offload()
            self._evaluate()
            self._dispatch_open_slots(force=True)
            while self._simulated_time < self.max_time and self._model_version < self.max_model_versions:
                outcome = self._wait_for_next_event()
                if outcome is None:
                    break
                self._process_outcome(outcome, accept_update=True)
                if (
                    self._simulated_time < self.max_time
                    and self._model_version < self.max_model_versions
                    and self._num_open_slots >= self.min_open_slots
                ):
                    self._dispatch_open_slots()

            # Reaching the simulated-time/model-version budget does not cancel
            # RPCs already executing.  Stop launching work, drain those calls,
            # and explicitly release their offloaded payloads.  This prevents
            # worker/callback state from leaking into shutdown.
            pending_count = len(self._pending_assignments)
            in_flight_count = len(self._physical_jobs)
            if pending_count or in_flight_count:
                self.logger.info(
                    f"canceling {pending_count} pending assignments and draining {in_flight_count} Collab calls"
                )
            self._pending_assignments.clear()
            while self._physical_jobs:
                self._record_physical_outcome(self._wait_for_any_outcome(), dispatch_pending=False)
            self._clear_completed_outcomes()
            self._clear_active_jobs()
            self._event_heap.clear()

            if not self._history or self._history[-1]["model_version"] != self._model_version:
                self._evaluate(final=True)
            else:
                self._history[-1]["final"] = True
                with open(
                    os.path.join(self._run_dir, "accuracy_history.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(self._history, stream, indent=2)
            results = {
                "method": self.method.value,
                "num_active_jobs": self.num_active_jobs,
                "buffer_size": self.buffer_size,
                "min_open_slots": self.min_open_slots,
                "in_time": self.in_time,
                "server_lr": self.server_lr,
                "fedrevive_mode": self.fedrevive_mode.value,
                "dfkd": (
                    {
                        "generator_steps": self._dfkd_config.generator_steps,
                        "generation_interval": self._dfkd_config.generation_interval,
                        "kd_iterations": self._dfkd_config.kd_iterations,
                        "teacher_buffer_size": self._dfkd_config.teacher_buffer_size,
                        "class_proportion_source": (
                            "two-upload-proxy"
                            if self.fedrevive_mode is FedReviveMode.PAPER_ALIGNED
                            else "prepared-label-histogram"
                        ),
                        "proxy_num_uploads": self._dfkd_config.proxy_num_uploads,
                        "proxy_temperature": self._dfkd_config.proxy_temperature,
                        "proxy_batch_size": self._dfkd_config.proxy_batch_size,
                    }
                    if self.method is Method.FEDREVIVE
                    else None
                ),
                "model_version": self._model_version,
                "simulated_time": self._simulated_time,
                "history": self._history,
            }
            with open(os.path.join(self._run_dir, "results.json"), "w", encoding="utf-8") as stream:
                json.dump(results, stream, indent=2)
            final_dir = os.path.join(self._run_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            torch.save(self._global_model, os.path.join(final_dir, "global_model.pt"))
            return self._global_model
        finally:
            # This also covers aborts and exceptions during DFKD or evaluation.
            self._clear_completed_outcomes()
            self._clear_active_jobs()
            self._stop_call_watchers()
            cleanup_tensor_disk_offload(
                engine=collab.fl_ctx.get_engine(),
                context=self._tensor_disk_offload_context,
            )
            if self._snapshot_store:
                self._snapshot_store.cleanup()
            if self._writer:
                self._writer.close()

    def _dispatch_open_slots(self, force: bool = False):
        """Create logical assignments when the configured O boundary is met.

        K is the logical concurrency limit, while O controls redistribution:
        O=1 refills immediately (FedBuff/asynchronous), and O=K waits for the
        cohort (FedAvg/synchronous).  Since a newly dispatched group shares one
        immutable snapshot, this boundary directly determines future update
        staleness and must not depend on physical worker availability.
        """

        if not self._available_logical or self._num_open_slots == 0:
            return
        if not force and self._num_open_slots < self.min_open_slots:
            return
        assignments = min(self._num_open_slots, len(self._available_logical))
        self._num_open_slots -= assignments

        # Persist once per global version and share a reference among all jobs
        # created from it.  Most of these K jobs will be pending on the small
        # physical pool or waiting for a later simulated upload event, so their
        # exact base weights belong on disk rather than in anonymous RAM.
        snapshot = self._snapshot_store.retain(self._model_version, self._global_model, assignments)
        logical_indices = np.atleast_1d(
            self._runtime_rng.choice(len(self._available_logical), size=assignments, replace=False)
        ).tolist()
        logical_names = [self._available_logical[int(index)] for index in logical_indices]
        for logical_index in sorted((int(index) for index in logical_indices), reverse=True):
            del self._available_logical[logical_index]

        for logical_name in logical_names:
            assignment_id = self._next_assignment_id
            self._next_assignment_id += 1
            train_seed = self.run_seed * 1_000_000 + assignment_id + 1
            job = _ActiveJob(
                logical_name=logical_name,
                assignment_id=assignment_id,
                model_version=self._model_version,
                base_snapshot=snapshot,
                train_seed=train_seed,
            )
            self._active_jobs[assignment_id] = job
            self._pending_assignments.append(assignment_id)
            download = float(self._manifest["runtime_profiles"][logical_name]["download_mean"])
            heapq.heappush(
                self._event_heap,
                _SimEvent(self._simulated_time + download, "download", assignment_id),
            )
        self._dispatch_pending_assignments()

    def _dispatch_pending_assignments(self):
        """Map queued logical jobs onto the bounded physical worker pool."""

        assignments_by_snapshot = {}
        while self._available_physical and self._pending_assignments:
            physical_name = self._available_physical.pop(0)
            assignment_id = self._pending_assignments.popleft()
            job = self._active_jobs[assignment_id]
            self._physical_jobs[physical_name] = assignment_id
            snapshot_key = (job.model_version, job.base_snapshot.path)
            assignments_by_snapshot.setdefault(snapshot_key, []).append((physical_name, assignment_id, job))

        for assignments in assignments_by_snapshot.values():
            physical_names = [physical_name for physical_name, _, _ in assignments]
            assignment_by_physical = {physical_name: assignment_id for physical_name, assignment_id, _ in assignments}
            logical_by_physical = {physical_name: job.logical_name for physical_name, _, job in assignments}
            train_seeds = {physical_name: job.train_seed for physical_name, _, job in assignments}
            representative = assignments[0][2]
            # collab.get_clients(...).train(...) invokes the @collab.publish
            # method on the named sites.  blocking=False is essential: waiting
            # for a group here would impose host completion order on the
            # simulated asynchronous sequence.  The callback receives each
            # successful result independently; the result iterator reports any
            # failures from the same group call.
            group = collab.get_clients(physical_names)
            # Load only the snapshot needed by the physical sites selected in
            # this call.  mmap-backed tensors remain file pages while Collab
            # serializes them; _ActiveJob continues to retain only the path.
            base_model = self._snapshot_store.load(representative.base_snapshot)
            results = group(
                blocking=False,
                timeout=self.call_timeout,
                parallel=self.max_parallel,
                process_resp_cb=self._accept_train_result,
            ).train(
                assignment_by_physical,
                representative.model_version,
                base_model,
                logical_by_physical,
                train_seeds,
            )
            # The Group dispatch owns its call arguments after train() returns;
            # dropping this local reference avoids extending the mapping's
            # lifetime.  A fixed pool consumes ResultQueues and reports errors.
            del base_model
            self._call_watch_queue.put((results, assignment_by_physical))

    def _accept_train_result(self, group_call_context, result):
        """Transfer a successful callback result to the scheduler thread."""

        physical_name = group_call_context.target_name.split(".", 1)[0]
        assignment_id = int(result[2]["assignment_id"])
        self._outcomes.put(_ClientOutcome(physical_name=physical_name, assignment_id=assignment_id, result=result))
        # Returning None avoids retaining or transforming a second model-sized
        # callback result.  The scheduler is the sole owner after queue.put().
        return None

    def _watch_call_failures(self, results, assignment_by_physical):
        for _ in results:
            pass
        for physical_name, error in results.failures.items():
            self._outcomes.put(
                _ClientOutcome(
                    physical_name=physical_name,
                    assignment_id=assignment_by_physical[physical_name],
                    error=error,
                )
            )

    def _wait_for_any_outcome(self):
        while True:
            try:
                return self._outcomes.get(timeout=1.0)
            except queue.Empty:
                if collab.is_aborted:
                    raise RuntimeError("FedRevive run aborted while waiting for a client result")

    def _wait_for_next_event(self):
        """Return the next upload in simulated order, regardless of RPC order.

        If its physical result is not ready, consume arbitrary completed RPCs
        into _completed_outcomes until it is.  Early results remain lazy and
        disk-backed.  Thus slow host scheduling changes wall-clock duration but
        cannot change client participation, staleness, or FedBuff membership.
        """

        while self._event_heap:
            event = self._event_heap[0]
            assignment_id = event.assignment_id
            job = self._active_jobs[assignment_id]
            if event.phase == "download":
                heapq.heappop(self._event_heap)
                self._simulated_time = event.finish_time
                heapq.heappush(
                    self._event_heap,
                    _SimEvent(
                        event.finish_time + self._sample_train_duration(job.logical_name),
                        "train",
                        assignment_id,
                    ),
                )
                if self._simulated_time >= self.max_time:
                    return None
                continue
            if event.phase == "train":
                heapq.heappop(self._event_heap)
                self._simulated_time = event.finish_time
                # The reference makes a logical client eligible again after
                # local training, before its upload completes.  Releasing it at
                # physical RPC completion would couple selection to host speed.
                if not job.logical_released:
                    self._available_logical.append(job.logical_name)
                    self._available_logical.sort(key=lambda name: int(name.split("-", 1)[1]))
                    job.logical_released = True
                job.finish_time = event.finish_time + self._sample_upload_duration(job.logical_name)
                heapq.heappush(
                    self._event_heap,
                    _SimEvent(job.finish_time, "upload", assignment_id),
                )
                if self._simulated_time >= self.max_time:
                    return None
                continue
            # Upload is the only phase that yields a model update.  It may have
            # completed physically much earlier and waited on disk.
            outcome = self._completed_outcomes.pop(assignment_id, None)
            if outcome is not None:
                heapq.heappop(self._event_heap)
                self._simulated_time = event.finish_time
                return outcome
            self._record_physical_outcome(self._wait_for_any_outcome())
        raise RuntimeError("No scheduled event remains")

    def _record_physical_outcome(self, outcome: _ClientOutcome, dispatch_pending: bool = True):
        """Free a physical worker without accepting the logical update yet."""

        expected_assignment = self._physical_jobs.pop(outcome.physical_name, None)
        if expected_assignment is None:
            self.logger.warning(f"ignoring unexpected result from {outcome.physical_name}")
            return
        if outcome.assignment_id != expected_assignment:
            raise RuntimeError(
                f"physical worker {outcome.physical_name} returned assignment {outcome.assignment_id}; "
                f"expected {expected_assignment}"
            )
        self._available_physical.append(outcome.physical_name)
        self._completed_outcomes[expected_assignment] = outcome
        # Reusing the physical process now improves throughput only.  The new
        # logical job already has its own seeded events and model snapshot, so
        # this cannot reorder accepted updates.
        if dispatch_pending:
            self._dispatch_pending_assignments()

    def _process_outcome(self, outcome: _ClientOutcome, accept_update: bool):
        """Materialize, consume, and release the logically next client result."""

        job = self._active_jobs.pop(outcome.assignment_id, None)
        if job is None:
            self.logger.warning(f"ignoring unexpected assignment {outcome.assignment_id}")
            return
        try:
            if not job.logical_released:
                self._available_logical.append(job.logical_name)
                self._available_logical.sort(key=lambda name: int(name.split("-", 1)[1]))
            self._num_open_slots += 1
            if not accept_update:
                self._release_outcome_result(outcome)
                return
            if outcome.error is not None:
                self.logger.warning(f"assignment {job.assignment_id} failed: {outcome.error}")
                return
            if outcome.result is None:
                self.logger.warning(f"assignment {job.assignment_id} returned no result")
                return

            # Materialize one client model at a time.  Remove the lazy payload
            # from the outcome before loading it, then unlink backing files
            # immediately.  This keeps the arbitrary-order completion queue on
            # disk even when host workers run far ahead of simulated uploads.
            result = outcome.result
            outcome.result = None
            try:
                updated_model, metrics, metadata = self._materialize_result(result)
            finally:
                if isinstance(result, (tuple, list)) and result and isinstance(result[0], LazyTensorDict):
                    result[0].cleanup()
                del result
            if (
                int(metadata["assignment_id"]) != job.assignment_id
                or int(metadata["model_version"]) != job.model_version
                or metadata["logical_name"] != job.logical_name
            ):
                raise RuntimeError(f"assignment metadata mismatch for {job.assignment_id}")

            staleness = self._model_version - job.model_version
            distilled_update = None
            dfkd_metrics = None
            if self.method is Method.FEDREVIVE:
                if self.fedrevive_mode is FedReviveMode.PAPER_ALIGNED:
                    proxy = self._proxy_estimator.estimate(job.logical_name, updated_model)
                    # If the same logical client still has an older model in
                    # the c=8 teacher buffer, its class weights are client
                    # state and must see the new two-upload running average.
                    self._teacher_buffer.update_class_proxy(job.logical_name, proxy)
                else:
                    proxy = torch.tensor(
                        self._manifest["class_proportions"][job.logical_name],
                        dtype=torch.float32,
                    )
                self._teacher_buffer.add(updated_model, proxy, job.logical_name, job.model_version)
                beta = cosine_staleness_beta(staleness, 75)
                if self._model_version >= self._dfkd_config.freeze_versions and beta > 0:
                    distilled_update, dfkd_metrics = self._reviver.revive(
                        self._global_model,
                        list(self._teacher_buffer),
                        self._model_version,
                    )

            # Load the assignment base only at the delta boundary.  The file
            # may represent a much older version, but the reference count keeps
            # it available until this exact assignment is accepted.
            base_model = self._snapshot_store.load(job.base_snapshot)
            try:
                # This is the common arrival boundary for all three algorithms.
                # B decides when a global version is created; arrival order
                # decides which deltas enter that buffer, which is why faithfully
                # replaying the delay schedule matters for FedBuff.
                aggregation_result = process_update(
                    method=self.method,
                    global_model=self._global_model,
                    updated_model=updated_model,
                    base_model=base_model,
                    client_name=job.logical_name,
                    base_version=job.model_version,
                    current_version=self._model_version,
                    update_buffer=self._update_buffer,
                    distilled_update=distilled_update,
                    max_staleness=75,
                    buffer_size=self.buffer_size,
                    server_lr=self.server_lr,
                )
            finally:
                del base_model
            self.logger.info(
                f"[{collab.call_info}] accepted assignment={job.assignment_id} logical={job.logical_name} "
                f"base={job.model_version} staleness={staleness} time={self._simulated_time:.2f} "
                f"buffer={len(self._update_buffer)}/{self.buffer_size} metrics={metrics} dfkd={dfkd_metrics}"
            )
            if aggregation_result is None:
                return
            self._global_model = aggregation_result.global_model
            self._model_version += 1
            if self._model_version % self.eval_interval == 0:
                self._evaluate()
            gc.collect()
        finally:
            # Release exactly once for success, client failure, cancellation,
            # or an exception while materializing/processing the update.  This
            # is what makes disk consumption follow live logical assignments.
            self._snapshot_store.release(job.base_snapshot)
