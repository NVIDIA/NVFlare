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

from typing import Any

from .av_model import AVModel
from .simple_shareable_generator import SimpleShareableGenerator


class AVShareableGenerator(SimpleShareableGenerator):
    def model_to_trainable(self, model_obj: Any) -> (dict, dict):
        assert isinstance(model_obj, AVModel)

        # Note: the weights returned must be serializable since it will be sent to clients
        return AVModel.serialize_layers(model_obj.free_layers), model_obj.meta

    def apply_weights_to_model(self, model_obj: Any, weights: Any, meta: dict) -> Any:
        # the "weights" is received from client. it has to be deserialized before processing
        assert isinstance(model_obj, AVModel)
        layers = AVModel.deserialize_layers(weights)

        print(f"apply layers to model: {layers}")

        # process received layers
        # this should be done based on meta information.
        # for example, if the "weights" contains diff, you need to add it to the based model.
        model_obj.free_layers = layers
        model_obj.meta.update(meta)
        return model_obj
