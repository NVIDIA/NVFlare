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

from nvflare.app_common.app_defined.shareable_generator import AppDefinedShareableGenerator

from .av_model import META_IS_DIFF, AVModel


class AVShareableGenerator(AppDefinedShareableGenerator):
    def model_to_trainable(self, model_obj: Any) -> (Any, dict):
        if not isinstance(model_obj, AVModel):
            raise ValueError(f"model object must be AVModel but got {type(model_obj)}")

        # only send free layers!
        return AVModel({}, {}, model_obj.free_layers), model_obj.meta

    def apply_weights_to_model(self, model_obj: Any, weights: Any, meta: dict) -> Any:
        if not isinstance(model_obj, AVModel):
            raise ValueError(f"model object must be AVModel but got {type(model_obj)}")

        if not isinstance(weights, AVModel):
            raise ValueError(f"weights object must be AVModel but got {type(weights)}")

        layers = weights.free_layers

        self.info(f"apply layers to model: {layers=}, {meta=}")

        # process received layers
        # this should be done based on meta information.
        # for example, if the "weights" contains diff, you need to add it to the based model.
        if not meta.get(META_IS_DIFF):
            model_obj.free_layers = layers
        else:
            # apply diffs
            free_layers = model_obj.free_layers
            meta.pop(META_IS_DIFF)
            for k, v in free_layers.items():
                for i, w in enumerate(v):
                    v[i] += layers[k][i]

        model_obj.meta.update(meta)
        self.info(f"aggregated full model: {model_obj.free_layers}")
        return model_obj
