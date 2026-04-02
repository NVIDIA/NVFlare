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

"""Base abstractions for synthetic data providers.

Defines the generic interface that all synthetic data providers must implement,
along with the protocol and type alias used to describe per-attribute value
generator callables.
"""

from abc import ABC, abstractmethod


class SyntheticDataProvider[T](ABC):
    """Abstract base class for synthetic data providers.

    Subclasses wrap a concrete data source (e.g. Faker, an RNG) and expose it
    through `provide`, returning domain-specific synthetic data of type `T`.
    """

    def __init__(self):
        ...

    @abstractmethod
    def provide(self) -> T:
        """Return a single synthetic data sample of type `T`."""
        raise NotImplementedError("Subclasses must implement the provide method.")
