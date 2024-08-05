# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Union


class FedObject(ABC):
    """Interface for objects with external resources used in FedJob API."""

    @abstractmethod
    def get_resources(self) -> Union[str, List[str]]:
        """Get resources (filenames or directories) to be included in job custom folder.

        Returns:
            path or list of paths of resources.
        """
        pass
