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

from abc import ABC, abstractmethod
from typing import List, Tuple


class EngineSpec(ABC):
    @abstractmethod
    def validate_targets(self, target_names: List[str]) -> Tuple[List, List[str]]:
        """Validate specified target names.

        Args:
            target_names: list of names to be validated

        Returns: a list of validate targets  and a list of invalid target names

        """
        pass
