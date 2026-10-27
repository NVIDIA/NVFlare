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

"""PyTorch model operations for buffered async training."""

import torch


class PTAsyncModelAdapter:
    """Keep detached CPU snapshots; average floating entries only.

    Non-floating buffers (e.g. BatchNorm counters) stay at the current global
    value. Supply another adapter for algorithms needing different buffer rules.
    """

    def snapshot(self, model):
        return {name: value.detach().cpu().clone() for name, value in model.items()}

    def validate(self, model, reference):
        if model.keys() != reference.keys():
            raise ValueError("model keys do not match reference")
        for name, value in model.items():
            base = reference[name]
            if not isinstance(value, torch.Tensor) or value.shape != base.shape or value.dtype != base.dtype:
                raise ValueError(f"incompatible model entry: {name}")
            if value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"model entry must be finite detached CPU data: {name}")

    def difference(self, model, base):
        return {name: value.detach().cpu() - base[name] for name, value in model.items() if value.is_floating_point()}

    def apply(self, model, delta, scale):
        result = self.snapshot(model)
        for name, value in delta.items():
            result[name].add_(value, alpha=scale)
        self.validate(result, model)
        return result
