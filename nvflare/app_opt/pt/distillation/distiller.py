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

"""Backend-independent supplied-data distillation contract."""

from typing import Protocol


class Distiller(Protocol):
    """A caller-owned distillation session, independent of its training backend.

    Applications bind backend configuration in a factory and depend on these
    three methods only. Construction defines teacher combination, temperature,
    reduction, device and batch interpretation explicitly. Native KD averages
    per-teacher losses; a mean-logit algorithm must supply one ensemble teacher.

    Built-in implementations consume FP32 inputs in one pass. Their metrics are
    loss (sample-weighted mean of the configured scalar batch loss), steps
    (optimizer updates), and samples (input rows). They use one optimizer update
    per batch; gradient accumulation is not implicit. Earlier updates remain
    applied if a later batch fails. Call close in a finally block on both success
    and failure. Sessions are not thread-safe.

    Student snapshots are detached, independent CPU state dictionaries compatible
    with the application's original student. Snapshotting does not finalize the
    session. Backend checkpoints, if provided, are a separate capability: model,
    optimizer, configuration and RNG portability across libraries is not promised.
    Teachers supplied by the caller must retain their state, mode and gradients.
    Backend-owned teacher copies may be frozen, wrapped and moved.

    Synthesis is a separate component producing batches. This same-batch contract
    does not implement FedRevive's independently sampled per-teacher KD policy.
    """

    def fit(self, batches) -> dict:
        """Consume supplied batches and return loss/steps/samples metrics."""
        ...

    def export_student_state(self) -> dict:
        """Return an owned student-only CPU snapshot without ending training."""
        ...

    def close(self) -> None:
        """Idempotently release owned hooks/resources; further training/export fails."""
        ...
