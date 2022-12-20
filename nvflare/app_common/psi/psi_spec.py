# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class PSI(FLComponent, ABC):
    """
    PSI interface is intended for end-user interface to
    get intersect without knowing the details of PSI algorithms, which will be delegated to the PSIHandler.
    """

    def __init__(self, psi_handler_id: str):
        super().__init__()
        self.psi_handler_id = psi_handler_id
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    @abstractmethod
    def load_items(self) -> List[str]:
        pass

    def save(self, intersections: List[str]):
        pass
