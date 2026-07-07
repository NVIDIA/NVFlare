# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# The SCAFFOLD-related functions are based on https://github.com/Xtra-Computing/NIID-Bench

# MIT License
#
# Copyright (c) 2021 Yiqun Diao, Qinbin Li
#
# Copyright (c) 2020 International Business Machines
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import weakref
from collections.abc import Mapping

import torch
from torch.optim import Optimizer


def get_lr_values(optimizer: Optimizer):
    """
    This function is used to get the learning rates of the optimizer.
    """
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]


class _ControlState:
    """A lightweight state-dict-compatible container for CPU control tensors."""

    def __init__(self, values):
        self._values = values

    def state_dict(self):
        return dict(self._values)

    def load_state_dict(self, values):
        self._values = dict(values)

    def to(self, device):
        self._values = {key: value.to(device) for key, value in self._values.items()}
        return self


class PTScaffoldHelper(object):
    """Helper to be used with SCAFFOLD components.
    Implements the functions used for the algorithm proposed in
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    (https://arxiv.org/abs/1910.06378) using PyTorch.
    SCAFFOLD-related functions are based on https://github.com/Xtra-Computing/NIID-Bench.
    See also Li et al. "Federated Learning on Non-IID Data Silos: An Experimental Study"
    (https://arxiv.org/abs/2102.02079).

    Control corrections and deltas apply only to trainable named parameters. Model buffers
    remain part of the regular model update and are not returned as control deltas. Persistent
    controls are stored on CPU, while one combined correction is materialized on the active
    parameter device during a training round.
    """

    def __init__(self):
        # SCAFFOLD control terms
        self.cnt = 0
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None
        self.device = None
        self._model_ref = None
        self._control_correction = {}
        self._previous_trainable_keys = None
        self._round_trainable_keys = None

    def init(self, model):
        self._prepare_model(model)
        parameters = dict(model.named_parameters())
        if not parameters:
            raise RuntimeError("PTScaffoldHelper requires a model with parameters.")

        self.c_global = _ControlState(self._zero_controls(parameters))
        self.c_local = _ControlState(self._zero_controls(parameters))
        self.device = next(iter(parameters.values())).device

    def _prepare_model(self, model):
        """Retain the active model without copying it and reconcile parameter control keys."""
        self._model_ref = weakref.ref(model)
        if self.c_global is None:
            return

        parameters = dict(model.named_parameters())
        if parameters:
            self.device = next(iter(parameters.values())).device
        global_controls = self.c_global.state_dict()
        local_controls = self.c_local.state_dict()
        reconciled_global = {}
        reconciled_local = {}
        for key, parameter in parameters.items():
            expected = parameter.detach().cpu()
            old_global = global_controls.get(key)
            old_local = local_controls.get(key)
            if old_global is None or tuple(old_global.shape) != tuple(expected.shape):
                old_global = torch.zeros_like(expected)
            else:
                old_global = old_global.to(dtype=expected.dtype, device="cpu")
            if old_local is None or tuple(old_local.shape) != tuple(expected.shape):
                old_local = torch.zeros_like(expected)
            else:
                old_local = old_local.to(dtype=expected.dtype, device="cpu")
            reconciled_global[key] = old_global
            reconciled_local[key] = old_local
        self.c_global.load_state_dict(reconciled_global)
        self.c_local.load_state_dict(reconciled_local)

    @staticmethod
    def _zero_controls(parameters):
        return {key: torch.zeros_like(parameter.detach(), device="cpu") for key, parameter in parameters.items()}

    def _get_model(self):
        model = self._model_ref() if self._model_ref is not None else None
        if model is None:
            raise RuntimeError("PTScaffoldHelper's initialized model is no longer available.")
        return model

    @staticmethod
    def _trainable_keys(model):
        return {key for key, parameter in model.named_parameters() if parameter.requires_grad}

    def get_params(self):
        self.cnt = 0
        self.c_delta_para = None
        self._control_correction = {}
        model = self._get_model()
        trainable_keys = self._trainable_keys(model)
        if self._previous_trainable_keys is not None:
            newly_trainable = trainable_keys - self._previous_trainable_keys
            if newly_trainable:
                local_controls = self.c_local.state_dict()
                parameters = dict(model.named_parameters())
                for key in newly_trainable:
                    local_controls[key] = torch.zeros_like(parameters[key].detach(), device="cpu")
                self.c_local.load_state_dict(local_controls)
        self._previous_trainable_keys = trainable_keys
        self._round_trainable_keys = trainable_keys
        # Adapted from https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L371
        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
        return c_global_para, c_local_para

    def _validate_trainable_keys(self, model):
        current_keys = self._trainable_keys(model)
        if current_keys != self._round_trainable_keys:
            added = sorted(current_keys - self._round_trainable_keys)
            removed = sorted(self._round_trainable_keys - current_keys)
            raise RuntimeError(
                "PTScaffoldHelper does not support changing requires_grad during a training round. "
                f"Newly trainable parameters: {added[:5]}; newly frozen parameters: {removed[:5]}."
            )

    def _get_correction(self, key, parameter, c_global_para, c_local_para):
        correction = self._control_correction.get(key)
        if correction is None or correction.device != parameter.device or correction.dtype != parameter.dtype:
            global_control = torch.as_tensor(c_global_para[key], dtype=parameter.dtype, device=parameter.device)
            local_control = torch.as_tensor(c_local_para[key], dtype=parameter.dtype, device=parameter.device)
            correction = global_control - local_control
            self._control_correction[key] = correction
        return correction

    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        # Update model using scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L391
        # SCAFFOLD controls apply to trainable parameters only. Model buffers such as
        # BatchNorm running statistics remain part of the regular model update.
        self._validate_trainable_keys(model)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                correction = self._get_correction(key, parameter, c_global_para, c_local_para)
                parameter.add_(correction, alpha=-curr_lr)

        self.cnt += 1

    def terms_update(self, model, curr_lr, c_global_para, c_local_para, model_global):
        # Update the local scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L403

        self._validate_trainable_keys(model)
        if self.cnt <= 0:
            raise RuntimeError("PTScaffoldHelper.terms_update requires at least one completed model_update call.")

        c_new_para = self.c_local.state_dict()
        self.c_delta_para = {}
        global_model_para = model_global.state_dict()
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                global_value = global_model_para[key].to(parameter)
                correction = self._get_correction(key, parameter, c_global_para, c_local_para)
                new_local = -correction + (global_value - parameter) / (self.cnt * curr_lr)
                c_new_para[key] = new_local.detach().cpu()
                old_local = torch.as_tensor(c_local_para[key], dtype=c_new_para[key].dtype, device="cpu")
                c_delta = c_new_para[key] - old_local
                if c_delta.dtype == torch.bfloat16:
                    c_delta = c_delta.to(torch.float32)
                self.c_delta_para[key] = c_delta.numpy()
        self.c_local.load_state_dict(c_new_para)
        self._control_correction = {}

    def load_global_controls(self, weights):
        if not isinstance(weights, Mapping) or not weights:
            raise RuntimeError("SCAFFOLD global controls must be a non-empty mapping.")

        model = self._get_model()
        parameters = dict(model.named_parameters())
        normalized = {}
        matched_keys = []
        for key, parameter in parameters.items():
            expected = parameter.detach().cpu()
            if key not in weights:
                normalized[key] = torch.zeros_like(expected)
                continue
            try:
                value = torch.as_tensor(weights[key])
            except (TypeError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to convert SCAFFOLD global control '{key}' to a tensor: {e}") from e
            if tuple(value.shape) != tuple(expected.shape):
                raise RuntimeError(
                    f"Invalid SCAFFOLD global control '{key}': expected shape {tuple(expected.shape)}, "
                    f"received {tuple(value.shape)}."
                )
            normalized[key] = value.to(dtype=expected.dtype, device="cpu")
            matched_keys.append(key)

        if not matched_keys:
            raise RuntimeError(
                "SCAFFOLD global controls do not match any named parameters in the local model. "
                f"Local parameter keys include {sorted(parameters)[:5]}; "
                f"received control keys include {sorted(weights)[:5]}."
            )
        self.c_global.load_state_dict(normalized)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")
        return self.c_delta_para
