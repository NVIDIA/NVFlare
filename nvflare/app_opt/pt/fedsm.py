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

"""PyTorch components and client helpers for personalized learning with FedSM."""

import copy
import os
import time
from numbers import Number
from typing import Dict, List, Optional

import numpy as np
import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg


class FedSMConstants:
    """Names used in the FedSM model bundle and client contract."""

    GLOBAL_MODEL = "global_model"
    PERSONAL_MODEL = "personal_model"
    PERSONAL_MODELS = "personal_models"
    SELECTOR_MODEL = "selector_model"
    SELECTOR_OPTIMIZER = "selector_optimizer"
    TARGET_ID = "fedsm_target_id"
    SELECTOR_LABEL = "fedsm_selector_label"


class PTFedSMHelper:
    """Load and return the multi-model bundle expected by ``FedSMRecipe``.

    The user remains responsible for the task-specific training loops. Call
    :meth:`load_bundle` before training, then :meth:`build_result` after training.
    """

    def __init__(self, global_model, personal_model, selector_model, selector_optimizer=None):
        for name, model in (
            ("global_model", global_model),
            ("personal_model", personal_model),
            ("selector_model", selector_model),
        ):
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f"{name} must be torch.nn.Module, got {type(model).__name__}")
        self.global_model = global_model
        self.personal_model = personal_model
        self.selector_model = selector_model
        self.selector_optimizer = selector_optimizer
        self._global_before = None
        self._selector_before = None

    def load_bundle(self, incoming: FLModel, client_name: Optional[str] = None):
        params = incoming.params or {}
        target = incoming.meta.get(FedSMConstants.TARGET_ID)
        if client_name is not None and target != client_name:
            raise ValueError(f"FedSM bundle targets {target!r}, but this client is {client_name!r}")
        self.global_model.load_state_dict(params[FedSMConstants.GLOBAL_MODEL])
        self.personal_model.load_state_dict(params[FedSMConstants.PERSONAL_MODEL])
        self.selector_model.load_state_dict(params[FedSMConstants.SELECTOR_MODEL])
        self._global_before = _to_cpu_copy(self.global_model.state_dict())
        self._selector_before = _to_cpu_copy(self.selector_model.state_dict())

        optimizer_state = params.get(FedSMConstants.SELECTOR_OPTIMIZER)
        if self.selector_optimizer is not None and optimizer_state:
            current_state = self.selector_optimizer.state_dict()
            current_state["state"] = optimizer_state
            self.selector_optimizer.load_state_dict(current_state)
        return incoming.meta[FedSMConstants.SELECTOR_LABEL]

    def build_result(self, num_steps: int, metrics: Optional[Dict] = None) -> FLModel:
        if self._global_before is None or self._selector_before is None:
            raise RuntimeError("load_bundle() must be called before build_result()")
        params = {
            FedSMConstants.GLOBAL_MODEL: self._state_diff(self._global_before, self.global_model.state_dict()),
            FedSMConstants.PERSONAL_MODEL: _to_cpu_copy(self.personal_model.state_dict()),
            FedSMConstants.SELECTOR_MODEL: self._state_diff(self._selector_before, self.selector_model.state_dict()),
        }
        if self.selector_optimizer is not None:
            params[FedSMConstants.SELECTOR_OPTIMIZER] = _to_cpu_copy(
                self.selector_optimizer.state_dict().get("state", {})
            )
        return FLModel(
            params=params,
            params_type=ParamsType.FULL,
            metrics=metrics,
            meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: num_steps},
        )

    @staticmethod
    def _state_diff(before: Dict, after: Dict) -> Dict:
        return {name: _to_cpu_copy(after[name]) - before[name] for name in before if name in after}


def _to_cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_copy(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _weighted_average(values: List, weights: List[float]):
    first = values[0]
    if isinstance(first, dict):
        common_keys = set(first)
        for value in values[1:]:
            common_keys.intersection_update(value)
        return {
            key: _weighted_average([value[key] for value in values], weights) for key in sorted(common_keys, key=str)
        }
    if isinstance(first, torch.Tensor):
        if not (first.is_floating_point() or first.is_complex()):
            averaged = torch.zeros_like(first, device="cpu", dtype=torch.float64)
            for value, weight in zip(values, weights):
                averaged.add_(torch.as_tensor(value, device="cpu", dtype=torch.float64), alpha=weight)
            return averaged.round().to(dtype=first.dtype)
        result = torch.zeros_like(first, device="cpu")
        for value, weight in zip(values, weights):
            result.add_(torch.as_tensor(value, device="cpu", dtype=result.dtype), alpha=weight)
        return result
    if isinstance(first, np.ndarray):
        return sum(weight * np.asarray(value) for value, weight in zip(values, weights))
    if isinstance(first, Number):
        return sum(weight * value for value, weight in zip(values, weights))
    raise TypeError(f"FedSM cannot average values of type {type(first).__name__}")


def _add_state_diff(state: Dict, state_diff: Dict) -> Dict:
    updated = _to_cpu_copy(state)
    for name, value in state_diff.items():
        if name not in updated:
            continue
        updated[name] = updated[name] + value
    return updated


class FedSMModelAggregator(ModelAggregator):
    """Aggregate global, personalized, and selector model updates for FedSM."""

    def __init__(self, soft_pull_lambda: float = 0.7):
        super().__init__()
        if not 0.0 <= soft_pull_lambda <= 1.0:
            raise ValueError(f"soft_pull_lambda must be in [0, 1], got {soft_pull_lambda}")
        self.soft_pull_lambda = soft_pull_lambda
        self._results: Dict[str, FLModel] = {}

    def reset_stats(self):
        self._results = {}

    def accept_model(self, model: FLModel):
        client_name = (model.meta or {}).get("client_name")
        if not client_name:
            raise ValueError("FedSM client result is missing FLModel.meta['client_name']")
        if model.params_type != ParamsType.FULL:
            raise ValueError(
                f"FedSM requires ParamsType.FULL bundle results, got {model.params_type} from {client_name!r}"
            )
        required = {
            FedSMConstants.GLOBAL_MODEL,
            FedSMConstants.PERSONAL_MODEL,
            FedSMConstants.SELECTOR_MODEL,
        }
        missing = required.difference(model.params or {})
        if missing:
            raise ValueError(f"FedSM client {client_name!r} result is missing bundle entries: {sorted(missing)}")
        if client_name in self._results:
            raise ValueError(f"FedSM received more than one result from client {client_name!r} in the same round")
        self._results[client_name] = model

    def aggregate_model(self) -> FLModel:
        if not self._results:
            raise ValueError("FedSM cannot aggregate an empty result set")

        clients = sorted(self._results)
        raw_weights = [
            float((self._results[client].meta or {}).get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)) for client in clients
        ]
        total = sum(raw_weights)
        weights = [weight / total for weight in raw_weights] if total > 0 else [1.0 / len(clients)] * len(clients)

        global_update = _weighted_average(
            [self._results[client].params[FedSMConstants.GLOBAL_MODEL] for client in clients],
            weights,
        )
        selector_update = _weighted_average(
            [self._results[client].params[FedSMConstants.SELECTOR_MODEL] for client in clients],
            weights,
        )

        personal_updates = {}
        if len(clients) == 1:
            only_client = clients[0]
            personal_updates[only_client] = _to_cpu_copy(
                self._results[only_client].params[FedSMConstants.PERSONAL_MODEL]
            )
        else:
            for target in clients:
                target_weights = [
                    self.soft_pull_lambda if source == target else (1.0 - self.soft_pull_lambda) / (len(clients) - 1)
                    for source in clients
                ]
                personal_updates[target] = _weighted_average(
                    [self._results[source].params[FedSMConstants.PERSONAL_MODEL] for source in clients],
                    target_weights,
                )

        bundle = {
            FedSMConstants.GLOBAL_MODEL: global_update,
            FedSMConstants.PERSONAL_MODELS: personal_updates,
            FedSMConstants.SELECTOR_MODEL: selector_update,
        }
        optimizer_states = [self._results[client].params.get(FedSMConstants.SELECTOR_OPTIMIZER) for client in clients]
        if all(state is not None for state in optimizer_states):
            bundle[FedSMConstants.SELECTOR_OPTIMIZER] = _weighted_average(optimizer_states, weights)

        return FLModel(
            params=bundle,
            params_type=ParamsType.FULL,
            current_round=self._results[clients[0]].current_round,
            meta={"nr_aggregated": len(clients)},
        )


class PTFedSMModelPersistor(ModelPersistor):
    """Persist the complete FedSM model bundle in a PyTorch checkpoint."""

    def __init__(
        self,
        model,
        selector_model,
        client_ids: List[str],
        source_ckpt_file_full_name: Optional[str] = None,
        global_model_file_name: str = "FL_fedsm_model.pt",
    ):
        super().__init__()
        self.model = model
        self.selector_model = selector_model
        self.client_ids = list(client_ids)
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        self.global_model_file_name = global_model_file_name
        self._ckpt_save_path = None

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def _initialize(self, fl_ctx: FLContext):
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        output_dir = os.path.join(app_root, log_dir) if log_dir else app_root
        os.makedirs(output_dir, exist_ok=True)
        self._ckpt_save_path = os.path.join(output_dir, self.global_model_file_name)
        self.model = self._resolve_model(self.model, "model", fl_ctx)
        self.selector_model = self._resolve_model(self.selector_model, "selector_model", fl_ctx)
        from nvflare.app_opt.pt.decomposers import TensorDecomposer
        from nvflare.fuel.utils import fobs

        fobs.register(TensorDecomposer)

    def _resolve_model(self, value, name: str, fl_ctx: FLContext):
        if isinstance(value, dict):
            from nvflare.fuel.utils.class_utils import instantiate_class

            value = instantiate_class(value.get("path"), value.get("args", {}))
        elif isinstance(value, str):
            value = fl_ctx.get_engine().get_component(value)
        if not isinstance(value, torch.nn.Module):
            raise TypeError(f"{name} must resolve to torch.nn.Module, got {type(value).__name__}")
        return value

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        checkpoint = self._load_checkpoint(fl_ctx)
        if checkpoint and "model_bundle" in checkpoint:
            return make_model_learnable(_to_cpu_copy(checkpoint["model_bundle"]), checkpoint.get("meta", {}))

        if checkpoint:
            global_state = checkpoint.get("model", checkpoint)
        else:
            global_state = self.model.state_dict()
        global_state = _to_cpu_copy(global_state)
        selector_state = _to_cpu_copy(self.selector_model.state_dict())
        bundle = {
            FedSMConstants.GLOBAL_MODEL: global_state,
            FedSMConstants.PERSONAL_MODELS: {client_id: _to_cpu_copy(global_state) for client_id in self.client_ids},
            FedSMConstants.SELECTOR_MODEL: selector_state,
            FedSMConstants.SELECTOR_OPTIMIZER: {},
        }
        return make_model_learnable(bundle, {})

    def _load_checkpoint(self, fl_ctx: FLContext):
        if not self.source_ckpt_file_full_name:
            return None
        path = self.source_ckpt_file_full_name
        if not os.path.isabs(path):
            app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
            path = os.path.join(app_root, WorkspaceConstants.CUSTOM_FOLDER_NAME, path)
        if not os.path.exists(path):
            raise ValueError(f"FedSM source checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        if self._ckpt_save_path is None:
            self._initialize(fl_ctx)
        torch.save(
            {
                "model_bundle": _to_cpu_copy(model[ModelLearnableKey.WEIGHTS]),
                "meta": copy.deepcopy(model[ModelLearnableKey.META]),
            },
            self._ckpt_save_path,
        )


class FedSM(BaseFedAvg):
    """Controller for the multi-model FedSM training lifecycle."""

    def __init__(
        self,
        *args,
        client_id_label_mapping: Dict[str, int],
        aggregator: Optional[FedSMModelAggregator] = None,
        task_name: str = AppConstants.TASK_TRAIN,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client_id_label_mapping = dict(client_id_label_mapping)
        self.aggregator = aggregator or FedSMModelAggregator()
        self.task_name = task_name

    def run(self):
        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"FedSM round {self.current_round} started")
            model.current_round = self.current_round
            self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
            self.fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)
            self.event(AppEventType.ROUND_STARTED)
            clients = self.sample_clients(self.num_clients)
            missing = [client for client in clients if client not in self.client_id_label_mapping]
            if missing:
                raise ValueError(f"FedSM has no selector labels for clients: {missing}")

            results = []
            for client in clients:
                request = self._make_client_model(model, client)
                self.send_model(
                    task_name=self.task_name,
                    targets=[client],
                    data=request,
                    callback=results.append,
                )

            while self.get_num_standing_tasks():
                if self.abort_signal.triggered:
                    return
                time.sleep(self._task_check_period)
            if len(results) != len(clients):
                raise RuntimeError(f"FedSM received {len(results)} of {len(clients)} expected client results")

            aggregated = self.aggregate(results, aggregate_fn=self._aggregate_results)
            self.event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE)
            model.params = self._update_bundle(model.params, aggregated.params)
            model.meta = aggregated.meta
            model.metrics = aggregated.metrics
            self.fl_ctx.set_prop(
                AppConstants.GLOBAL_MODEL,
                make_model_learnable(model.params, model.meta),
                private=True,
                sticky=True,
            )
            self.event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE)
            self.save_model(model)
            self.event(AppEventType.ROUND_DONE)
            self.info(f"FedSM round {self.current_round} finished")
            self._maybe_cleanup_memory()

    def _make_client_model(self, model: FLModel, client: str) -> FLModel:
        bundle = model.params
        params = {
            FedSMConstants.GLOBAL_MODEL: _to_cpu_copy(bundle[FedSMConstants.GLOBAL_MODEL]),
            FedSMConstants.PERSONAL_MODEL: _to_cpu_copy(bundle[FedSMConstants.PERSONAL_MODELS][client]),
            FedSMConstants.SELECTOR_MODEL: _to_cpu_copy(bundle[FedSMConstants.SELECTOR_MODEL]),
            FedSMConstants.SELECTOR_OPTIMIZER: _to_cpu_copy(bundle.get(FedSMConstants.SELECTOR_OPTIMIZER, {})),
        }
        return FLModel(
            params=params,
            params_type=ParamsType.FULL,
            current_round=self.current_round,
            total_rounds=self.num_rounds,
            meta={
                FedSMConstants.TARGET_ID: client,
                FedSMConstants.SELECTOR_LABEL: self.client_id_label_mapping[client],
            },
        )

    def _aggregate_results(self, results: List[FLModel]) -> FLModel:
        self.aggregator.reset_stats()
        for result in results:
            self.aggregator.accept_model(result)
        return self.aggregator.aggregate_model()

    @staticmethod
    def _update_bundle(bundle: Dict, update: Dict) -> Dict:
        updated = _to_cpu_copy(bundle)
        updated[FedSMConstants.GLOBAL_MODEL] = _add_state_diff(
            updated[FedSMConstants.GLOBAL_MODEL], update[FedSMConstants.GLOBAL_MODEL]
        )
        updated[FedSMConstants.SELECTOR_MODEL] = _add_state_diff(
            updated[FedSMConstants.SELECTOR_MODEL], update[FedSMConstants.SELECTOR_MODEL]
        )
        for client, personal_model in update[FedSMConstants.PERSONAL_MODELS].items():
            updated[FedSMConstants.PERSONAL_MODELS][client] = _to_cpu_copy(personal_model)
        if FedSMConstants.SELECTOR_OPTIMIZER in update:
            updated[FedSMConstants.SELECTOR_OPTIMIZER] = _to_cpu_copy(update[FedSMConstants.SELECTOR_OPTIMIZER])
        return updated
