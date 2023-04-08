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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.model_desc import ModelDescriptor

from .learnable_persistor import LearnablePersistor
from .model import ModelLearnable
from .persistor_filter import PersistorFilter


class ModelPersistor(LearnablePersistor, ABC):
    def __init__(self, filter_id: str = None):
        """Abstract class.
        Implementations will need to implement the `load_model()` and `save_model()`
        methods to persist & load the current ModelLearnable.
            Args:
                filter_id: Optional string that defines a filter component that is applied to prepare the model to be saved,
                    e.g. for serialization of custom Python objects.

        """
        super().__init__()
        self.filter_id = filter_id

    def load(self, fl_ctx: FLContext) -> ModelLearnable:
        learnable = self.load_model(fl_ctx)
        if self.filter_id:
            _filter = fl_ctx.get_engine().get_component(self.filter_id)
            if not isinstance(_filter, PersistorFilter):
                raise ValueError(f"Expected filter to be of type `PersistorFilter` but got {type(filter)}")
            learnable = _filter.process_post_load(learnable=learnable, fl_ctx=fl_ctx)
        return learnable

    def save(self, learnable: ModelLearnable, fl_ctx: FLContext):
        if self.filter_id:
            _filter = fl_ctx.get_engine().get_component(self.filter_id)
            if not isinstance(_filter, PersistorFilter):
                raise ValueError(f"Expected filter to be of type `PersistorFilter` but got {type(filter)}")
            learnable = _filter.process_pre_save(learnable=learnable, fl_ctx=fl_ctx)

        self.save_model(learnable, fl_ctx)

        if self.filter_id:
            _filter.process_post_save(learnable=learnable, fl_ctx=fl_ctx)

    def get(self, model_file, fl_ctx: FLContext) -> object:
        learnable = self.get_model(model_file, fl_ctx)

        if self.filter_id:
            _filter = fl_ctx.get_engine().get_component(self.filter_id)
            if not isinstance(_filter, PersistorFilter):
                raise ValueError(f"Expected filter to be of type `PersistorFilter` but got {type(filter)}")
            learnable = _filter.process_post_get(learnable=learnable, fl_ctx=fl_ctx)
        return learnable

    @abstractmethod
    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initialize and load the model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object

        """
        pass

    @abstractmethod
    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        """Persist the model object.

        Args:
            model: Model object to be saved
            fl_ctx: FLContext

        """
        pass

    def get_model_inventory(self, fl_ctx: FLContext) -> {str: ModelDescriptor}:
        """Get the model inventory of the ModelPersister.

        Args:
            fl_ctx: FLContext

        Returns: { model_kind: ModelDescriptor }

        """
        pass

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        pass
