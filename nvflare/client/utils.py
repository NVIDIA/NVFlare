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

from typing import Dict, Iterable

from nvflare.app_common.abstract.fl_model import FLModel

from .constants import CONST_ATTRS, ModelExchangeFormat


def get_meta_from_fl_model(fl_model: FLModel, attrs: Iterable[str]) -> Dict:
    """Get metadata from an FLModel object.

    Args:
        fl_model: an FLModel object.
        attrs: attributes to get from FLModel.

    Returns:
        A dictionary with attribute name as key and FLModel's attribute as value.
    """
    meta = {}
    for attr in attrs:
        if hasattr(fl_model, attr):
            meta[attr] = getattr(fl_model, attr)
        elif attr in fl_model.meta:
            meta[attr] = fl_model.meta[attr]
        else:
            raise RuntimeError(f"can't find attribute {attr} in fl_model.")
    return meta


def set_fl_model_with_meta(fl_model: FLModel, meta: Dict, attrs):
    """Sets FLModel attributes.

    Args:
        fl_model: an FLModel object.
        meta: a dict contains attributes.
        attrs: attributes to set.
    """
    for attr in attrs:
        setattr(fl_model, attr, meta[attr])
        meta.pop(attr)


def copy_fl_model_attributes(src: FLModel, dst: FLModel, attrs=CONST_ATTRS):
    """Copies FLModel attributes from source to destination.

    Args:
        src: source FLModel object.
        dst: destination FLModel object.
        attrs: attributes to copy.
    """
    for attr in attrs:
        setattr(dst, attr, getattr(src, attr))


def numerical_params_diff(original: Dict, new: Dict) -> Dict:
    """Calculates the numerical parameter difference.

    Args:
        original: A dict of numerical values.
        new: A dict of numerical values.

    Returns:
        A dict with same key as original dict,
        value are the difference between original and new.
    """
    diff_dict = {}
    for k in original:
        if k not in new:
            continue
        diff_dict[k] = new[k] - original[k]
    return diff_dict


DIFF_FUNCS = {ModelExchangeFormat.PYTORCH: numerical_params_diff, ModelExchangeFormat.NUMPY: numerical_params_diff}
