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

"""Capacity-bounded synthetic tensors with caller-selected sampling."""

import torch


class SyntheticPool:
    """Keep the most recent max_samples input/target pairs on CPU.

    Inputs have shape [samples, ...]; optional targets can be class labels,
    regression values or other tensors sharing that leading dimension.
    Adding detaches/copies tensors. Shapes, dtypes and target presence must stay
    consistent until the pool is empty. No image/class assumptions are made.
    """

    def __init__(self, max_samples: int):
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.max_samples = max_samples
        self._inputs = None
        self._targets = None

    def __len__(self):
        return 0 if self._inputs is None else len(self._inputs)

    def add(self, inputs: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Retain the newest inputs/targets, copying their detached values to CPU."""
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or not len(inputs):
            raise ValueError("inputs must be a nonempty batched tensor")
        if targets is not None and (
            not isinstance(targets, torch.Tensor) or targets.ndim < 1 or len(targets) != len(inputs)
        ):
            raise ValueError("targets must have the same leading dimension as inputs")
        if self._inputs is not None:
            if inputs.shape[1:] != self._inputs.shape[1:] or inputs.dtype != self._inputs.dtype:
                raise ValueError("input shape/dtype must match the existing pool")
            if (targets is None) != (self._targets is None):
                raise ValueError("target presence must match the existing pool")
            if targets is not None and (
                targets.shape[1:] != self._targets.shape[1:] or targets.dtype != self._targets.dtype
            ):
                raise ValueError("target shape/dtype must match the existing pool")
        # Bound incoming copies too; adding a large batch does not copy all of it.
        inputs = inputs[-self.max_samples :].detach().cpu().clone()
        targets = None if targets is None else targets[-self.max_samples :].detach().cpu().clone()
        if self._inputs is not None:
            inputs = torch.cat((self._inputs, inputs))[-self.max_samples :].clone()
            if targets is not None:
                targets = torch.cat((self._targets, targets))[-self.max_samples :].clone()
        self._inputs, self._targets = inputs, targets

    def sample(self, batch_size: int, sampler=None, generator: torch.Generator | None = None):
        """Return owned (inputs, targets) tensors, sampling with replacement.

        sampler(inputs, targets, batch_size, generator) may return CPU int64
        indices to replace uniform sampling. Its input tensors are borrowed
        read-only. generator, when supplied, must be a CPU generator and is
        caller-owned; omitting it uses the global CPU RNG.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not len(self):
            raise ValueError("cannot sample an empty synthetic pool")
        indices = (
            torch.randint(len(self), (batch_size,), generator=generator, device="cpu")
            if sampler is None
            else sampler(self._inputs, self._targets, batch_size, generator)
        )
        if (
            not isinstance(indices, torch.Tensor)
            or indices.shape != (batch_size,)
            or indices.dtype != torch.long
            or indices.device.type != "cpu"
            or (indices < 0).any()
            or (indices >= len(self)).any()
        ):
            raise ValueError("sampler must return batch_size valid CPU int64 indices")
        return self._inputs[indices], None if self._targets is None else self._targets[indices]

    def state_dict(self):
        """Return independent CPU copies. Sampling/RNG policy is caller-owned."""
        return {
            "max_samples": self.max_samples,
            "inputs": None if self._inputs is None else self._inputs.clone(),
            "targets": None if self._targets is None else self._targets.clone(),
        }

    def load_state_dict(self, state):
        """Restore the same capacity without silently truncating history."""
        if state["max_samples"] != self.max_samples:
            raise ValueError("checkpoint capacity does not match this pool")
        restored = SyntheticPool(self.max_samples)
        if state["inputs"] is not None:
            if len(state["inputs"]) > self.max_samples:
                raise ValueError("checkpoint exceeds pool capacity")
            restored.add(state["inputs"], state["targets"])
        elif state["targets"] is not None:
            raise ValueError("empty checkpoint cannot contain targets")
        self._inputs, self._targets = restored._inputs, restored._targets
