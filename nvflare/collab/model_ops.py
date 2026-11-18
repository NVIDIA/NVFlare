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

"""Framework-independent operations on dictionary model states."""

from typing import Protocol


class ModelOps(Protocol):
    """Arithmetic policy for model states; methods must not mutate inputs."""

    def snapshot(self, model: dict) -> dict:
        """Return an independent model state suitable for storage and transfer."""

    def validate(self, model: dict, reference: dict) -> None:
        """Check model keys, shapes, dtypes and values against a reference."""

    def difference(self, model: dict, base: dict) -> dict:
        """Return an owned delta for entries participating in aggregation."""

    def apply(self, model: dict, delta: dict, scale: float) -> dict:
        """Return a new model after applying a scaled delta."""
