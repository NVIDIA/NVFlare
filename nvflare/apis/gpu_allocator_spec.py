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

"""Cooperative GPU assignment, independent of process creation and job admission."""

from abc import ABC, abstractmethod


class GpuAllocationError(RuntimeError):
    """A GPU assignment or its release could not be confirmed."""


class GpuReservationSpec(ABC):
    """An accounting claim, without device-access or compute/memory isolation.

    The caller releases the claim only after its execution scope has settled.
    Disappearance of the caller alone does not establish settlement.
    """

    @property
    @abstractmethod
    def devices(self) -> tuple[str, ...]:
        """Assigned device identifiers suitable for CUDA_VISIBLE_DEVICES."""
        raise NotImplementedError()

    @abstractmethod
    def record_execution(self, execution_id: str):
        """Associate the physical execution with this claim for diagnostics."""
        raise NotImplementedError()

    @abstractmethod
    def release(self):
        """Release a settled claim, idempotently after a confirmed release."""
        raise NotImplementedError()


class GpuAllocatorSpec(ABC):
    """Assign devices from an administrator-configured inventory.

    Implementations must coordinate all participating callers that share devices.
    This is separate from ResourceManagerSpec's job-admission handshake, whose
    unallocated reservations expire. Neither contract enforces hardware isolation.
    """

    @abstractmethod
    def validate_host(self):
        """Validate the configured devices against the current host."""
        raise NotImplementedError()

    @abstractmethod
    def acquire(self, identity: tuple[str, ...], count: int) -> GpuReservationSpec:
        """Atomically claim all requested devices or raise GpuAllocationError.

        A successfully admitted identity must never start a second execution,
        including after a caller restart or after its claim has been released.
        """
        raise NotImplementedError()
