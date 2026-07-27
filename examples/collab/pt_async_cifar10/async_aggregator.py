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

"""Asynchronous server workflow using Collab in-time response callbacks."""

import gc
import os
import random
import threading
import time
from collections import defaultdict

import numpy as np
import torch
from model import add_update_to_params, get_model_params, load_model_params, reset_model_state, resnet18_local
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.fuel.utils.log_utils import get_obj_logger

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _AggrResult:
    def __init__(self):
        self.total_delta: dict[str, torch.Tensor] = {}
        self.count = 0
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.client_names: list[str] = []
        self.lock = threading.Lock()


class Cifar10AsyncAggregator:
    """Aggregate client models as their Collab responses arrive."""

    def __init__(
        self,
        data_root: str,
        logical_num_clients: int,
        num_rounds: int = 3,
        clients_per_round: int | None = None,
        min_response_clients: int | None = None,
        call_timeout: float = 3600.0,
        max_parallel: int = 0,
        device: str | None = None,
        eval_interval: int = 1,
        eval_batch_size: int = 300,
        server_lr: float = 1.0,
        setup_seed: int = 10,
        run_seed: int = 10,
    ):
        self.data_root = data_root
        self.logical_num_clients = logical_num_clients
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.min_response_clients = min_response_clients
        self.call_timeout = call_timeout
        self.max_parallel = max_parallel
        self.device_name = device
        self.eval_interval = max(1, eval_interval)
        self.eval_batch_size = max(1, eval_batch_size)
        self.server_lr = float(server_lr)
        self.setup_seed = setup_seed
        self.run_seed = run_seed
        self.logger = get_obj_logger(self)

        self._logical_client_names = [f"site-{index}" for index in range(1, logical_num_clients + 1)]
        self._current_assignment: dict[str, str] = {}
        self._round_base_model: dict[str, torch.Tensor] | None = None
        self._test_loader = None
        self._eval_model = None
        self._writer = None
        self._run_dir = None

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
            self._eval_model = resnet18_local()
        reset_model_state(self._eval_model, reset_norm_stats=True)
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
        return accuracy

    @collab.main
    def execute(self):
        self.logger.info(
            f"[{collab.call_info}] starting CIFAR-10 training for {self.num_rounds} rounds "
            f"with {self.logical_num_clients} logical clients"
        )
        seed_everything(self.setup_seed)
        self._init_outputs()
        initial_model = get_model_params(resnet18_local(), target_device="cpu")
        global_model = collab.get_prop(ContextKey.RESULT, initial_model)
        global_model = {name: value.detach().cpu().clone() for name, value in global_model.items()}

        try:
            self._evaluate_model(global_model, round_index=-1)
            seed_everything(self.run_seed)
            for round_index in range(self.num_rounds):
                round_started = time.time()
                global_model = self._do_one_round(round_index, global_model)
                if (round_index + 1) % self.eval_interval == 0 or round_index == self.num_rounds - 1:
                    self._evaluate_model(global_model, round_index)
                gc.collect()
                self.logger.info(
                    f"[{collab.call_info}] round {round_index + 1}/{self.num_rounds} completed "
                    f"in {time.time() - round_started:.3f}s"
                )

            final_dir = os.path.join(self._run_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            final_path = os.path.join(final_dir, "global_model.pt")
            torch.save(global_model, final_path)
            self.logger.info(f"Saved final model to {final_path}")
            return global_model
        finally:
            if self._writer:
                self._writer.close()

    def _do_one_round(self, round_index: int, global_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        all_clients = collab.clients
        physical_names = [client.name for client in all_clients]
        desired_clients = min(self.clients_per_round or len(all_clients), len(all_clients))
        if desired_clients < 1:
            raise RuntimeError("No clients are available for training")
        if desired_clients < len(all_clients):
            rng = np.random.RandomState(round_index)
            sampled_names = rng.choice(physical_names, size=desired_clients, replace=False).tolist()
            group = collab.get_clients(sampled_names)
        else:
            sampled_names = physical_names
            group = all_clients

        min_required = self.min_response_clients if self.min_response_clients is not None else desired_clients
        if min_required > desired_clients:
            raise ValueError(
                f"min_response_clients ({min_required}) cannot exceed selected clients ({desired_clients})"
            )

        logical_assignments = self._build_logical_assignments(round_index, sampled_names)
        self._current_assignment = logical_assignments
        self._round_base_model = global_model
        aggr_result = _AggrResult()

        self.logger.info(
            f"[{collab.call_info}] round {round_index + 1}: calling {desired_clients} clients; "
            f"minimum responses={min_required}; assignments={logical_assignments}"
        )
        call_started = time.time()
        results = group(
            blocking=True,
            timeout=self.call_timeout,
            parallel=self.max_parallel,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(round_index, global_model, logical_assignments)

        for client_name, error in results.failures.items():
            self.logger.warning(f"round {round_index + 1}: client {client_name} failed: {error}")
        if aggr_result.count < min_required:
            raise RuntimeError(
                f"Round {round_index + 1} received {aggr_result.count} successful responses; "
                f"{min_required} required"
            )

        result = self._finalize_aggregation(aggr_result)
        average_metrics = {name: sum(values) / len(values) for name, values in aggr_result.metrics.items() if values}
        self.logger.info(
            f"[{collab.call_info}] round {round_index + 1}: aggregated {aggr_result.count} responses "
            f"from {aggr_result.client_names}; metrics={average_metrics}; calls={time.time() - call_started:.3f}s"
        )

        round_dir = os.path.join(self._run_dir, "round_models")
        os.makedirs(round_dir, exist_ok=True)
        torch.save(result, os.path.join(round_dir, f"model_round_{round_index + 1}.pt"))
        self._round_base_model = None
        return result

    def _build_logical_assignments(self, round_index: int, physical_names: list[str]) -> dict[str, str]:
        if len(self._logical_client_names) < len(physical_names):
            raise ValueError(
                f"Only {len(self._logical_client_names)} prepared logical clients are available for "
                f"{len(physical_names)} physical clients"
            )
        rng = np.random.RandomState(round_index + 1234)
        logical_names = rng.choice(self._logical_client_names, size=len(physical_names), replace=False).tolist()
        return dict(zip(physical_names, logical_names))

    def _accept_train_result(self, group_call_context, result, aggr_result: _AggrResult):
        client_name = group_call_context.target_name.split(".", 1)[0]
        logical_client = self._current_assignment.get(client_name, client_name)
        if result is None:
            self.logger.warning(f"[{collab.call_info}] client {client_name} returned no update")
            return None

        self.logger.info(f"[{collab.call_info}] received update from {client_name} (logical={logical_client})")
        model, metrics = result
        with aggr_result.lock:
            aggr_result.client_names.append(client_name)
            for metric_name, metric_value in metrics.items():
                aggr_result.metrics[metric_name].append(metric_value)

            for name, value in model.items():
                value = value.detach().cpu()
                delta = value - self._round_base_model[name]
                if name in aggr_result.total_delta:
                    aggr_result.total_delta[name].add_(delta)
                else:
                    aggr_result.total_delta[name] = delta.clone()
            aggr_result.count += 1
        return None

    def _finalize_aggregation(self, aggr_result: _AggrResult) -> dict[str, torch.Tensor]:
        if not aggr_result.count or self._round_base_model is None:
            raise RuntimeError("Cannot aggregate an empty response buffer")
        average_update = {
            name: total_delta / aggr_result.count for name, total_delta in aggr_result.total_delta.items()
        }
        return add_update_to_params(self._round_base_model, average_update, scale=self.server_lr)
