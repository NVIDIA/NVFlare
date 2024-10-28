# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable


class ShareableGenerator(FLComponent, ABC):
    @abstractmethod
    def learnable_to_shareable(self, model: Learnable, fl_ctx: FLContext) -> Shareable:
        """Generate the initial Shareable from the Learnable object.

        Args:
            model: model object
            fl_ctx: FLContext

        Returns:
            shareable

        """
        pass

    @abstractmethod
    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        """Construct the Learnable object from Shareable.

        Args:
            shareable: shareable
            fl_ctx: FLContext

        Returns:
            model object

        """
        pass
