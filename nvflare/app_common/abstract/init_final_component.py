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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class InitFinalComponent(FLComponent, ABC):
    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        pass


class InitFinalArgsComponent(InitFinalComponent, ABC):
    @abstractmethod
    def initialize(self, fl_ctx: FLContext, **kwargs):
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        pass
