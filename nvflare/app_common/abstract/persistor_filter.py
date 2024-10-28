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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

from .model import Learnable


class PersistorFilter(FLComponent):
    def process_post_load(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object after it was loaded in ModelPersistor's `load()` call.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return learnable

    def process_pre_save(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object before its being saved in ModelPersistor's `save()` call.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return learnable

    def process_post_save(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object after it's being saved in ModelPersistor's `save()` call.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return learnable

    def process_post_get(self, learnable: Learnable, fl_ctx: FLContext) -> Learnable:
        """Filter process applied to the Learnable object after it was returned in ModelPersistor's `get()` call.

        Args:
            learnable: Learnable
            fl_ctx: FLContext

        Returns:
            a Learnable object
        """
        return learnable
