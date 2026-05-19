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

import copy
import inspect
from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import Optimizer

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper, get_lr_values
from nvflare.client.config import ConfigKey
from nvflare.fuel.utils.log_utils import get_obj_logger

PT_SCAFFOLD_AUTO_PATCH = "pt_scaffold_auto_patch"


class PTScaffoldAutoPatchManager:
    """Runtime hooks that make standard PyTorch Client API scripts SCAFFOLD-aware.

    This manager intentionally supports the common Client API pattern:
    ``flare.receive()``, ``model.load_state_dict(input_model.params)``, ``optimizer.step()``,
    and ``flare.send(output_model)``. Unsupported patterns fail with explicit errors instead of
    silently producing an invalid SCAFFOLD update.
    """

    def __init__(self):
        self._enabled = False
        self._original_load_state_dict = None
        self._original_optimizer_steps = {}
        self._original_helper_model_update = None
        self._original_helper_terms_update = None
        self._internal_helper_call = False
        self._suspend_model_load_detection = 0
        self.logger = get_obj_logger(self)

        self._helper = None
        self._reset_round_state(clear_helper=True)

    def enable(self):
        if self._enabled:
            return self

        self._enabled = True
        self._patch_load_state_dict()
        self._patch_optimizer_steps()
        self._patch_helper_methods()
        self.logger.info("Auto-SCAFFOLD PyTorch mode enabled for this client process.")
        return self

    def disable(self):
        if not self._enabled:
            self._reset_round_state(clear_helper=True)
            return

        if self._original_load_state_dict is not None:
            torch.nn.Module.load_state_dict = self._original_load_state_dict
            self._original_load_state_dict = None

        for optimizer_cls, original_step in self._original_optimizer_steps.items():
            optimizer_cls.step = original_step
        self._original_optimizer_steps = {}

        if self._original_helper_model_update is not None:
            PTScaffoldHelper.model_update = self._original_helper_model_update
            self._original_helper_model_update = None
        if self._original_helper_terms_update is not None:
            PTScaffoldHelper.terms_update = self._original_helper_terms_update
            self._original_helper_terms_update = None

        self._enabled = False
        self._reset_round_state(clear_helper=True)
        self.logger.info("Auto-SCAFFOLD PyTorch mode disabled for this client process.")

    def on_receive(self, input_model: Optional[FLModel], task_name: Optional[str], train_task_name: Optional[str]):
        if not self._enabled or input_model is None:
            return

        if task_name != train_task_name:
            self._reset_round_state(clear_helper=False)
            return

        if input_model.params is None:
            raise RuntimeError("Auto-SCAFFOLD requires the received training FLModel to contain params.")
        if not input_model.meta or AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL not in input_model.meta:
            raise RuntimeError(
                "Auto-SCAFFOLD requires received FLModel.meta to contain "
                f"'{AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL}'."
            )

        self._reset_round_state(clear_helper=False)
        self._active = True
        self._input_params = input_model.params
        self._global_ctrl_weights = input_model.meta[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL]
        self.logger.info(
            f"Auto-SCAFFOLD received training round {input_model.current_round}; "
            "waiting for model.load_state_dict(input_model.params)."
        )

    def on_model_load(self, model: torch.nn.Module, state_dict: Mapping[str, Any]):
        if (
            not self._enabled
            or not self._active
            or self._suspend_model_load_detection
            or not self._matches_input_params(state_dict)
        ):
            return

        if self._step_count > 0:
            raise RuntimeError("Auto-SCAFFOLD does not support re-loading the received model after optimizer steps.")
        if self._active_model is not None and self._active_model is not model:
            raise RuntimeError(
                "Auto-SCAFFOLD detected multiple models loaded from the received FLModel. "
                "Use manual PTScaffoldHelper mode for multi-model scripts."
            )

        self._active_model = model
        self._global_model = copy.deepcopy(model)
        self._global_model.eval()
        for param in self._global_model.parameters():
            param.requires_grad = False
        self.logger.info(f"Auto-SCAFFOLD detected model '{model.__class__.__name__}' for client-side correction.")

    def on_optimizer_step(self, optimizer: Optimizer):
        if not self._enabled or not self._active:
            return

        if self._active_model is None:
            return
        if not self._optimizer_updates_active_model(optimizer):
            return
        if not self._round_prepared:
            self._prepare_round_for_step()

        curr_lr = self._get_single_lr(optimizer)
        self._call_helper_method(
            self._original_helper_model_update,
            self._helper,
            model=self._active_model,
            curr_lr=curr_lr,
            c_global_para=self._c_global_para,
            c_local_para=self._c_local_para,
        )
        self._last_lr = curr_lr
        self._step_count += 1

    def on_send(self, output_model: FLModel) -> FLModel:
        if not self._enabled or not self._active:
            return output_model

        if output_model.params is None:
            raise RuntimeError("Auto-SCAFFOLD requires the training result FLModel to contain params.")

        output_model.meta = dict(output_model.meta) if output_model.meta else {}
        if AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta:
            raise RuntimeError(
                "Auto-SCAFFOLD is enabled, but the outgoing FLModel already contains "
                f"'{AlgorithmConstants.SCAFFOLD_CTRL_DIFF}'. Remove manual PTScaffoldHelper calls or disable "
                "auto_scaffold."
            )
        if self._active_model is None:
            raise RuntimeError(
                "Auto-SCAFFOLD did not detect model.load_state_dict(input_model.params) before flare.send()."
            )
        if self._step_count <= 0:
            raise RuntimeError("Auto-SCAFFOLD did not detect any optimizer.step() calls for the loaded model.")
        if self._last_lr is None:
            raise RuntimeError("Auto-SCAFFOLD could not determine the optimizer learning rate.")

        self._move_global_model_to_active_device()
        self._call_helper_method(
            self._original_helper_terms_update,
            self._helper,
            model=self._active_model,
            curr_lr=self._last_lr,
            c_global_para=self._c_global_para,
            c_local_para=self._c_local_para,
            model_global=self._global_model,
        )
        output_model.meta.setdefault(FLMetaKey.NUM_STEPS_CURRENT_ROUND, self._step_count)
        output_model.meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF] = self._helper.get_delta_controls()
        self.logger.info(
            f"Auto-SCAFFOLD added '{AlgorithmConstants.SCAFFOLD_CTRL_DIFF}' after "
            f"{self._step_count} optimizer step(s)."
        )

        self._reset_round_state(clear_helper=False)
        return output_model

    def _reset_round_state(self, clear_helper: bool):
        self._active = False
        self._input_params = None
        self._global_ctrl_weights = None
        self._active_model = None
        self._global_model = None
        self._c_global_para = None
        self._c_local_para = None
        self._round_prepared = False
        self._step_count = 0
        self._last_lr = None
        if clear_helper:
            self._helper = None

    def _patch_load_state_dict(self):
        manager = self
        self._original_load_state_dict = torch.nn.Module.load_state_dict
        original_load_state_dict = self._original_load_state_dict

        def patched_load_state_dict(module, state_dict, *args, **kwargs):
            result = original_load_state_dict(module, state_dict, *args, **kwargs)
            manager.on_model_load(module, state_dict)
            return result

        torch.nn.Module.load_state_dict = patched_load_state_dict

    def _patch_optimizer_steps(self):
        manager = self
        for optimizer_cls in self._optimizer_classes():
            original_step = getattr(optimizer_cls, "step", None)
            if original_step is None or optimizer_cls in self._original_optimizer_steps:
                continue

            def make_patched_step(original):
                def patched_step(optimizer, *args, **kwargs):
                    result = original(optimizer, *args, **kwargs)
                    manager.on_optimizer_step(optimizer)
                    return result

                return patched_step

            self._original_optimizer_steps[optimizer_cls] = original_step
            optimizer_cls.step = make_patched_step(original_step)

    def _patch_helper_methods(self):
        manager = self
        self._original_helper_model_update = PTScaffoldHelper.model_update
        self._original_helper_terms_update = PTScaffoldHelper.terms_update

        def patched_model_update(helper, *args, **kwargs):
            manager._raise_manual_helper_call("model_update")
            return manager._original_helper_model_update(helper, *args, **kwargs)

        def patched_terms_update(helper, *args, **kwargs):
            manager._raise_manual_helper_call("terms_update")
            return manager._original_helper_terms_update(helper, *args, **kwargs)

        PTScaffoldHelper.model_update = patched_model_update
        PTScaffoldHelper.terms_update = patched_terms_update

    def _raise_manual_helper_call(self, method_name: str):
        if self._enabled and not self._internal_helper_call:
            raise RuntimeError(
                f"PTScaffoldHelper.{method_name}() cannot be called while auto_scaffold=True. "
                "Use manual SCAFFOLD mode or remove manual PTScaffoldHelper correction calls."
            )

        return None

    @staticmethod
    def _optimizer_classes():
        classes = {Optimizer}
        stack = [Optimizer]
        while stack:
            optimizer_cls = stack.pop()
            for subclass in optimizer_cls.__subclasses__():
                if subclass not in classes:
                    classes.add(subclass)
                    stack.append(subclass)

        for value in vars(torch.optim).values():
            if inspect.isclass(value) and issubclass(value, Optimizer):
                classes.add(value)

        return classes

    def _matches_input_params(self, state_dict: Mapping[str, Any]) -> bool:
        if state_dict is self._input_params:
            return True
        if not isinstance(state_dict, Mapping) or not isinstance(self._input_params, Mapping):
            return False
        if set(state_dict.keys()) != set(self._input_params.keys()):
            return False

        for name, value in state_dict.items():
            if self._shape_of(value) != self._shape_of(self._input_params[name]):
                return False
        return True

    @staticmethod
    def _shape_of(value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        return tuple(torch.as_tensor(value).shape)

    def _prepare_round_for_step(self):
        if self._global_model is None:
            raise RuntimeError("Auto-SCAFFOLD could not preserve the received global model.")

        self._move_global_model_to_active_device()
        if self._helper is None:
            self._helper = PTScaffoldHelper()
            self._call_helper_method(self._helper.init, model=self._active_model)
        else:
            device = self._get_model_device(self._active_model)
            self._helper.c_global.to(device)
            self._helper.c_local.to(device)

        global_ctrl_state = self._to_model_state_dict(self._global_ctrl_weights, self._active_model)
        self._call_helper_method(self._helper.load_global_controls, weights=global_ctrl_state)
        self._c_global_para, self._c_local_para = self._helper.get_params()
        self._round_prepared = True

    def _move_global_model_to_active_device(self):
        device = self._get_model_device(self._active_model)
        self._global_model.to(device)

    @staticmethod
    def _get_model_device(model: torch.nn.Module):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_model_state_dict(self, weights: Mapping[str, Any], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        model_state = model.state_dict()
        device = self._get_model_device(model)
        state_dict = {}
        for name, value in weights.items():
            target = model_state.get(name)
            tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if target is not None:
                tensor = tensor.to(device=device, dtype=target.dtype)
            else:
                tensor = tensor.to(device=device)
            state_dict[name] = tensor
        return state_dict

    def _optimizer_updates_active_model(self, optimizer: Optimizer) -> bool:
        model_param_ids = {id(param) for param in self._active_model.parameters()}
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if id(param) in model_param_ids:
                    return True
        return False

    @staticmethod
    def _get_single_lr(optimizer: Optimizer) -> float:
        lr_values = get_lr_values(optimizer)
        if not lr_values:
            raise RuntimeError("Auto-SCAFFOLD could not determine optimizer learning rate.")

        first_lr = lr_values[0]
        for lr in lr_values[1:]:
            if lr != first_lr:
                raise RuntimeError(
                    "Auto-SCAFFOLD does not support optimizers with multiple different learning rates. "
                    "Use manual PTScaffoldHelper mode for this training loop."
                )
        return float(first_lr)

    def _call_helper_method(self, method, *args, **kwargs):
        self._internal_helper_call = True
        self._suspend_model_load_detection += 1
        try:
            return method(*args, **kwargs)
        finally:
            self._suspend_model_load_detection -= 1
            self._internal_helper_call = False


_MANAGER = PTScaffoldAutoPatchManager()


def get_pt_scaffold_auto_patch_manager() -> PTScaffoldAutoPatchManager:
    return _MANAGER


def maybe_enable_pt_scaffold_auto_patch(client_config):
    task_exchange_config = client_config.get_config().get(ConfigKey.TASK_EXCHANGE, {})
    if task_exchange_config.get(PT_SCAFFOLD_AUTO_PATCH):
        return _MANAGER.enable()
    return None


def disable_pt_scaffold_auto_patch():
    _MANAGER.disable()
