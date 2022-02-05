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

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
    validate_model_learnable,
)
from nvflare.app_common.app_constant import ModelFormat


def feed_vars(model: nn.Module, model_params):
    """Feed variable values from model_params to pytorch state_dict.

    Args:
        model (nn.Module): the local pytorch model
        model_params: a ModelData message

    Returns:
        a list of params and a dictionary of vars to params
    """
    _logger = logging.getLogger("AssignVariables")
    _logger.debug("AssignVariables...")

    to_assign = []
    n_ext = len(model_params)
    _logger.debug(f"n_ext {n_ext}")

    local_var_dict = model.state_dict()
    for var_name in local_var_dict:
        try:
            if var_name in tuple(model_params):
                nd = model_params[var_name]
                to_assign.append(nd)
                local_var_dict[var_name] = torch.as_tensor(
                    nd
                )  # update local state dict TODO: enable setting of datatype
        except Exception as e:
            print("pt_feed_vars Exception:", str(e))
            raise RuntimeError(str(e))

    _logger.debug("Updated local variables to be assigned.")

    n_assign = len(to_assign)
    _logger.info(f"Vars {n_ext} of {n_assign} assigned.")
    return to_assign, local_var_dict


class PTModelPersistenceFormatManager(object):

    PERSISTENCE_KEY_MODEL = "model"
    PERSISTENCE_KEY_TRAIN_CONF = "train_conf"
    PERSISTENCE_KEY_META_PROPS = "meta_props"

    def __init__(self, data: dict, default_train_conf=None):
        """Manage the format for model persistence.

        Args:
            data (dict): either the dictionary mapping variables to values or a dict of dict.
            default_train_conf (dict, optional): configuration for train. Defaults to None.

        Raises:
            TypeError: when data is not a dictionary
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict but got {}".format(type(data)))

        self.var_dict = None
        self.meta = None
        self.train_conf = None
        self.other_props = {}  # other props from the original data that need to be kept

        if self.PERSISTENCE_KEY_MODEL not in data:
            # this is a simple weight dict
            self.var_dict = data
        else:
            # dict of dicts
            self.var_dict = data[self.PERSISTENCE_KEY_MODEL]
            self.meta = data.get(self.PERSISTENCE_KEY_META_PROPS, None)
            self.train_conf = data.get(self.PERSISTENCE_KEY_TRAIN_CONF, None)

            # we need to keep other props, if any, so they can be kept when persisted
            for k, v in data.items():
                if k not in [
                    self.PERSISTENCE_KEY_MODEL,
                    self.PERSISTENCE_KEY_META_PROPS,
                    self.PERSISTENCE_KEY_TRAIN_CONF,
                ]:
                    self.other_props[k] = v

        if not self.train_conf:
            self.train_conf = default_train_conf

    def _get_processed_vars(self) -> dict:
        if self.meta:
            return self.meta.get(MetaKey.PROCESSED_KEYS, {})
        else:
            return {}

    def to_model_learnable(self, exclude_vars) -> ModelLearnable:
        processed_vars = self._get_processed_vars()

        weights = {}
        for k, v in self.var_dict.items():
            if exclude_vars and exclude_vars.search(k):
                continue

            is_processed = processed_vars.get(k, False)
            if is_processed:
                weights[k] = v
            else:
                weights[k] = v.cpu().numpy()

        return make_model_learnable(weights, self.meta)

    def to_persistence_dict(self) -> dict:
        processed_vars = self._get_processed_vars()
        weights_dict = OrderedDict()
        for k, v in self.var_dict.items():
            is_processed = processed_vars.get(k, False)
            if is_processed:
                weights_dict[k] = v
            else:
                weights_dict[k] = torch.as_tensor(v)

        # always use complex format for saving
        persistence_dict = OrderedDict()
        persistence_dict[self.PERSISTENCE_KEY_MODEL] = weights_dict
        if self.meta:
            persistence_dict[self.PERSISTENCE_KEY_META_PROPS] = self.meta
        if self.train_conf:
            persistence_dict[self.PERSISTENCE_KEY_TRAIN_CONF] = self.train_conf
        if self.other_props:
            for k, v in self.other_props.items():
                persistence_dict[k] = v
        return persistence_dict

    def update(self, ml: ModelLearnable):
        """Update the persistence data with the learned values.

        Args:
            ml (ModelLearnable): updated information to be merged into existing ModelLearnable
        """
        err = validate_model_learnable(ml)
        if err:
            raise ValueError(err)
        self.meta = ml.get(ModelLearnableKey.META, None)

        # update with value of the model learnable
        # note that the original weights that are not learned are still kept!
        learned_weights = ml.get(ModelLearnableKey.WEIGHTS, {})
        for k, v in learned_weights.items():
            self.var_dict[k] = v

    def get_persist_model_format(self):
        return ModelFormat.PT_CHECKPOINT
