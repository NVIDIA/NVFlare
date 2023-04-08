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

"""The Learnable in the deep learning domain is usually called Model by researchers.

This import simply lets you call the Learnable 'Model'.
Model Learnable is a dict that contains two items: weights and meta info
"""
from nvflare.apis.dxo import DXO, DataKind

from .learnable import Learnable


class ModelLearnableKey(object):
    WEIGHTS = "weights"
    META = "meta"


class ModelLearnable(Learnable):
    def is_empty(self):
        if self.get(ModelLearnableKey.WEIGHTS):
            return False
        else:
            return True


def validate_model_learnable(model_learnable: ModelLearnable) -> str:
    """Check whether the specified model is a valid Model Shareable.

    Args:
        model_learnable (ModelLearnable): model to be validated

    Returns:
        str: error text or empty string if no error
    """
    if not isinstance(model_learnable, ModelLearnable):
        return "invalid model learnable: expect Model type but got {}".format(type(model_learnable))

    if ModelLearnableKey.WEIGHTS not in model_learnable:
        return "invalid model learnable: missing weights"

    if ModelLearnableKey.META not in model_learnable:
        return "invalid model learnable: missing meta"

    return ""


def make_model_learnable(weights, meta_props) -> ModelLearnable:
    ml = ModelLearnable()
    ml[ModelLearnableKey.WEIGHTS] = weights
    ml[ModelLearnableKey.META] = meta_props
    return ml


def model_learnable_to_dxo(ml: ModelLearnable) -> DXO:
    err = validate_model_learnable(ml)
    if err:
        raise ValueError(err)

    return DXO(data_kind=DataKind.WEIGHTS, data=ml[ModelLearnableKey.WEIGHTS], meta=ml[ModelLearnableKey.META])
