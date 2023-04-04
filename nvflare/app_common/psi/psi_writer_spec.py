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
from typing import List

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class PSIWriter(FLComponent, ABC):
    """Interface for saving PSI intersection."""

    @abstractmethod
    def save(self, intersection: List[str], overwrite_existing: bool, fl_ctx: FLContext):
        """Saves PSI intersection.

        Args:
            intersection: (List[str]) - Intersection to be saved
            overwrite_existing: (bool) overwrite the existing one if true
            fl_ctx: (FLContext)

        """
        pass
