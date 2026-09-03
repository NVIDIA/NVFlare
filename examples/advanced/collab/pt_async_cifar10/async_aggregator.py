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

"""Online FedBuff server workflow built with nonblocking Collab calls."""

import gc
import json
import os
import queue
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from model import ModerateCNN, add_update_to_params, get_model_params, load_model_params
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.fuel.utils.log_utils import get_obj_logger

_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        ),
    ]
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class _ActiveJob:
    physical_name: str
    logical_name: str
    assignment_id: int
    model_version: int
    base_model: dict[str, torch.Tensor]


@dataclass
class _ClientOutcome:
    physical_name: str
    result: object = None
    error: object = None


@dataclass
class _BufferedUpdate:
    job: _ActiveJob
    delta: dict[str, torch.Tensor]
    metrics: dict[str, float]


class Cifar10AsyncAggregator:
    """Maintain active jobs and aggregate each full FedBuff update buffer."""

    def __init__(
        self,
        data_root: str,
        num_rounds: int = 100,
        num_active_jobs: int = 8,
        buffer_size: int = 4,
        min_open_slots: int = 1,
        call_timeout: float = 3600.0,
        device: str | None = None,
        eval_batch_size: int = 300,
        server_lr: float = 1.0,
        setup_seed: int = 10,
        run_seed: int = 10,
        checkpoint_interval: int = 0,
    ):
        if num_active_jobs < 1:
            raise ValueError("num_active_jobs must be >= 1")
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if not 1 <= min_open_slots <= num_active_jobs:
            raise ValueError("min_open_slots must be between 1 and num_active_jobs")

        self.data_root = data_root
        self.num_rounds = num_rounds
        self.num_active_jobs = num_active_jobs
        self.buffer_size = buffer_size
        self.min_open_slots = min_open_slots
        self.call_timeout = call_timeout
        self.device_name = device
        self.eval_batch_size = max(1, eval_batch_size)
        self.server_lr = float(server_lr)
        self.setup_seed = setup_seed
        self.run_seed = run_seed
        self.checkpoint_interval = max(0, checkpoint_interval)
        self.logger = get_obj_logger(self)

        self._outcomes = queue.Queue()
        self._active_jobs: dict[str, _ActiveJob] = {}
        self._available_clients: list[str] = []
        self._num_open_slots = 0
        self._update_buffer: list[_BufferedUpdate] = []
        self._next_assignment_id = 0
        self._model_version = 0
        self._rounds_completed = 0
        self._global_model: dict[str, torch.Tensor] | None = None
        self._test_loader = None
        self._eval_model = None
        self._writer = None
        self._run_dir = None
        self._last_round_time = None
        self._test_history = []
        self._eval_snapshot_dir = None

    def _new_model(self):
        return ModerateCNN()

    def _device(self) -> torch.device:
        return torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _init_outputs(self):
        self._run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        tensorboard_dir = os.path.join(self._run_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=tensorboard_dir)
        self.logger.info(f"TensorBoard logs: {tensorboard_dir}")

    def _init_test_loader(self):
        if self._test_loader is None:
            test_set = datasets.CIFAR10(
                root=self.data_root,
                train=False,
                download=False,
                transform=_TEST_TRANSFORM,
            )
            self._test_loader = DataLoader(test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0)

    def _evaluate_model(self, global_model: dict[str, torch.Tensor], round_index: int) -> float:
        self._init_test_loader()
        device = self._device()
        if self._eval_model is None:
            self._eval_model = self._new_model()
        load_model_params(self._eval_model, global_model, target_device=device)
        self._eval_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self._test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                predictions = self._eval_model(inputs).argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0
        display_round = "initial" if round_index < 0 else str(round_index + 1)
        self.logger.info(f"[{collab.call_info}] round {display_round}: test accuracy={accuracy:.4f}")
        self._writer.add_scalar("test/accuracy", accuracy, round_index + 1)
        self._writer.flush()
        self._test_history.append(
            {
                "aggregation": round_index + 1,
                "accepted_updates": max(0, round_index + 1) * self.buffer_size,
                "accuracy": accuracy,
            }
        )
        with open(os.path.join(self._run_dir, "accuracy_history.json"), "w") as history_file:
            json.dump(self._test_history, history_file, indent=2)
        return accuracy

    def _save_eval_snapshot(self) -> None:
        if self._eval_snapshot_dir is None:
            self._eval_snapshot_dir = os.path.join(self._run_dir, "eval_snapshots")
            os.makedirs(self._eval_snapshot_dir, exist_ok=True)
        torch.save(self._global_model, os.path.join(self._eval_snapshot_dir, f"model_version_{self._model_version}.pt"))

    def _evaluate_snapshots(self) -> None:
        self.logger.info(f"evaluating {self.num_rounds + 1} saved global model versions on the server")
        for model_version in range(self.num_rounds + 1):
            snapshot_path = os.path.join(self._eval_snapshot_dir, f"model_version_{model_version}.pt")
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            self._evaluate_model(snapshot, round_index=model_version - 1)

    # @collab.main marks the server-side workflow entry point. Nonblocking
    # client calls below remain active while this method advances FedBuff.
    @collab.main
    def execute(self):
        self.logger.info(
            f"[{collab.call_info}] starting online CIFAR-10 FedBuff for {self.num_rounds} aggregations: "
            f"K={self.num_active_jobs}, B={self.buffer_size}, O={self.min_open_slots}"
        )
        seed_everything(self.setup_seed)
        self._init_outputs()
        initial_model = get_model_params(self._new_model(), target_device="cpu")
        global_model = collab.get_prop(ContextKey.RESULT, initial_model)
        self._global_model = {name: value.detach().cpu().clone() for name, value in global_model.items()}
        self._save_eval_snapshot()

        all_physical_names = [client.name for client in collab.clients]
        if self.num_active_jobs > len(all_physical_names):
            raise ValueError(
                f"num_active_jobs ({self.num_active_jobs}) exceeds available physical clients "
                f"({len(all_physical_names)})"
            )

        try:
            seed_everything(self.run_seed)
            self._available_clients = all_physical_names
            self._num_open_slots = self.num_active_jobs
            self._dispatch_open_slots(force=True)
            self._last_round_time = time.time()

            while self._rounds_completed < self.num_rounds:
                outcome = self._wait_for_outcome()
                aggregated = self._process_outcome(outcome, accept_update=True)

                # Aggregate first when both thresholds are reached. Jobs
                # dispatched here therefore receive the newly created version.
                if self._rounds_completed < self.num_rounds and self._num_open_slots >= self.min_open_slots:
                    self._dispatch_open_slots()
                if aggregated:
                    self._after_aggregation()

            # Calls already in flight cannot be cancelled through the public
            # Collab group API. Drain them without changing the final model.
            if self._active_jobs:
                self.logger.info(f"draining {len(self._active_jobs)} jobs still active after the final aggregation")
            while self._active_jobs:
                self._process_outcome(self._wait_for_outcome(), accept_update=False)

            self._evaluate_snapshots()

            final_dir = os.path.join(self._run_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            final_path = os.path.join(final_dir, "global_model.pt")
            torch.save(self._global_model, final_path)
            self.logger.info(f"Saved final model to {final_path}")
            return self._global_model
        finally:
            if self._writer:
                self._writer.close()

    def _dispatch_open_slots(self, force: bool = False) -> None:
        if not self._available_clients or self._num_open_slots == 0:
            return
        if not force and self._num_open_slots < self.min_open_slots:
            return

        target_updates = self.num_rounds * self.buffer_size
        accepted_or_buffered = self._rounds_completed * self.buffer_size + len(self._update_buffer)
        assignments_needed = target_updates - accepted_or_buffered - len(self._active_jobs)
        assignments_needed = min(assignments_needed, self._num_open_slots, len(self._available_clients))
        if assignments_needed <= 0:
            return
        physical_names = self._available_clients[:assignments_needed]
        self._available_clients = self._available_clients[assignments_needed:]
        self._num_open_slots -= len(physical_names)
        logical_assignments = physical_names
        logical_by_physical = dict(zip(physical_names, logical_assignments))
        assignment_ids = {}
        model_snapshot = {name: value.detach().clone() for name, value in self._global_model.items()}

        for physical_name in physical_names:
            assignment_id = self._next_assignment_id
            self._next_assignment_id += 1
            logical_name = logical_by_physical[physical_name]
            assignment_ids[physical_name] = assignment_id
            self._active_jobs[physical_name] = _ActiveJob(
                physical_name=physical_name,
                logical_name=logical_name,
                assignment_id=assignment_id,
                model_version=self._model_version,
                base_model=model_snapshot,
            )

        self.logger.info(
            f"[{collab.call_info}] dispatching model version {self._model_version} to "
            f"{physical_names}; assignments={logical_by_physical}"
        )
        group = collab.get_clients(physical_names)
        # blocking=False returns immediately. @collab.publish on the client
        # exposes train(), and each response is delivered to the callback.
        results = group(
            blocking=False,
            timeout=self.call_timeout,
            process_resp_cb=self._accept_train_result,
        ).train(
            assignment_ids,
            self._model_version,
            model_snapshot,
            logical_by_physical,
        )
        threading.Thread(
            target=self._watch_call_failures,
            args=(results,),
            daemon=True,
            name=f"fedbuff_dispatch_v{self._model_version}",
        ).start()

    def _accept_train_result(self, group_call_context, result):
        physical_name = group_call_context.target_name.split(".", 1)[0]
        self._outcomes.put(_ClientOutcome(physical_name=physical_name, result=result))
        return None

    def _watch_call_failures(self, results) -> None:
        # Iteration waits for every outcome from this nonblocking group call.
        # Successful values were already handled by the response callback.
        for _ in results:
            pass
        for physical_name, error in results.failures.items():
            self._outcomes.put(_ClientOutcome(physical_name=physical_name, error=error))

    def _wait_for_outcome(self) -> _ClientOutcome:
        while True:
            try:
                return self._outcomes.get(timeout=1.0)
            except queue.Empty:
                if collab.is_aborted:
                    raise RuntimeError("FedBuff run aborted while waiting for a client outcome")
                if not self._active_jobs:
                    raise RuntimeError("FedBuff has no active jobs and cannot make progress")

    def _process_outcome(self, outcome: _ClientOutcome, accept_update: bool) -> bool:
        job = self._active_jobs.pop(outcome.physical_name, None)
        if job is None:
            self.logger.warning(f"ignoring duplicate or unexpected outcome from {outcome.physical_name}")
            return False

        self._available_clients.append(job.physical_name)
        self._num_open_slots += 1
        if outcome.error is not None:
            self.logger.warning(
                f"assignment {job.assignment_id} failed on {job.physical_name} "
                f"(logical={job.logical_name}): {outcome.error}"
            )
            return False
        if outcome.result is None:
            self.logger.warning(f"assignment {job.assignment_id} returned no update from {job.physical_name}")
            return False
        if not accept_update:
            self.logger.info(f"discarding assignment {job.assignment_id} completed after the final aggregation")
            return False

        updated_model, metrics, metadata = outcome.result
        if int(metadata["assignment_id"]) != job.assignment_id or int(metadata["model_version"]) != job.model_version:
            raise RuntimeError(
                f"assignment metadata mismatch from {job.physical_name}: expected "
                f"({job.assignment_id}, {job.model_version}), got "
                f"({metadata['assignment_id']}, {metadata['model_version']})"
            )
        delta = {name: value.detach().cpu() - job.base_model[name] for name, value in updated_model.items()}
        self._update_buffer.append(_BufferedUpdate(job=job, delta=delta, metrics=metrics))
        staleness = self._model_version - job.model_version
        self.logger.info(
            f"[{collab.call_info}] accepted assignment {job.assignment_id} from {job.physical_name} "
            f"(logical={job.logical_name}, base_version={job.model_version}, staleness={staleness}); "
            f"buffer={len(self._update_buffer)}/{self.buffer_size}"
        )

        if len(self._update_buffer) < self.buffer_size:
            return False
        self._aggregate_buffer()
        return True

    def _aggregate_buffer(self) -> None:
        total_delta = {}
        metrics = defaultdict(list)
        weights = [float(update.metrics["num_steps"]) for update in self._update_buffer]
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("aggregation weights must sum to a positive value")
        for update, weight in zip(self._update_buffer, weights):
            for name, value in update.delta.items():
                if name in total_delta:
                    total_delta[name].add_(value, alpha=weight)
                else:
                    total_delta[name] = value.clone().mul_(weight)
            for metric_name, metric_value in update.metrics.items():
                metrics[metric_name].append(metric_value)

        average_update = {name: value / total_weight for name, value in total_delta.items()}
        self._global_model = add_update_to_params(self._global_model, average_update, scale=self.server_lr)
        client_names = [update.job.logical_name for update in self._update_buffer]
        average_metrics = {name: sum(values) / len(values) for name, values in metrics.items() if values}
        self._update_buffer.clear()
        self._model_version += 1
        self._rounds_completed += 1
        self.logger.info(
            f"[{collab.call_info}] created model version {self._model_version}/{self.num_rounds} "
            f"from {client_names}; metrics={average_metrics}"
        )

    def _after_aggregation(self) -> None:
        self._save_eval_snapshot()

        if self.checkpoint_interval and (
            self._rounds_completed % self.checkpoint_interval == 0 or self._rounds_completed == self.num_rounds
        ):
            round_dir = os.path.join(self._run_dir, "round_models")
            os.makedirs(round_dir, exist_ok=True)
            torch.save(self._global_model, os.path.join(round_dir, f"model_round_{self._rounds_completed}.pt"))
        gc.collect()
        self.logger.info(
            f"[{collab.call_info}] aggregation {self._rounds_completed}/{self.num_rounds} completed "
            f"in {time.time() - self._last_round_time:.3f}s"
        )
        self._last_round_time = time.time()
