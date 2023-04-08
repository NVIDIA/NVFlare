# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, model_learnable_to_dxo
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


def _get_xgboost_model_attr(xgb_model):
    num_parallel_tree = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"]
    )
    if "best_iteration" in xgb_model["learner"]["attributes"].keys():
        best_iteration = int(xgb_model["learner"]["attributes"]["best_iteration"])
    else:
        best_iteration = 1
    best_ntree_limit = int(xgb_model["learner"]["attributes"]["best_ntree_limit"])
    num_trees = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    return num_parallel_tree, best_iteration, best_ntree_limit, num_trees


def update_model(prev_model, model_update):
    if not prev_model:
        return model_update
    else:
        # Append all trees
        # get the parameters
        pre_num_parallel_tree, pre_best_iteration, pre_best_ntree_limit, pre_num_trees = _get_xgboost_model_attr(
            prev_model
        )
        cur_num_parallel_tree, add_best_iteration, add_best_ntree_limit, add_num_trees = _get_xgboost_model_attr(
            model_update
        )

        # check num_parallel_tree, should be consistent
        if cur_num_parallel_tree != pre_num_parallel_tree:
            raise ValueError(
                f"add_num_parallel_tree should not change, previous {pre_num_parallel_tree}, current {add_num_parallel_tree}"
            )
        prev_model["learner"]["attributes"]["best_iteration"] = str(pre_best_iteration + 1)
        prev_model["learner"]["attributes"]["best_ntree_limit"] = str(pre_best_ntree_limit + cur_num_parallel_tree)
        prev_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
            pre_num_trees + cur_num_parallel_tree
        )
        # append the new trees
        append_info = model_update["learner"]["gradient_booster"]["model"]["trees"]
        for tree_ct in range(cur_num_parallel_tree):
            append_info[tree_ct]["id"] = pre_num_trees + tree_ct
            prev_model["learner"]["gradient_booster"]["model"]["trees"].append(append_info[tree_ct])
            prev_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
        return prev_model


class XGBModelShareableGenerator(ShareableGenerator):
    def __init__(self):
        super().__init__()
        self.shareable = None

    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """

        if not self.shareable:
            # initialization or recovering from previous training
            model = model_learnable[ModelLearnableKey.WEIGHTS]
            if model:
                # recovering from previous run - distinguish between cyclic and bagging modes as
                # global model format is different
                if isinstance(model, dict):
                    # bagging mode
                    serialized_model = bytearray(json.dumps(model), "utf-8")
                else:
                    # cyclic mode, model should be serialized already
                    serialized_model = model
                dxo = DXO(data_kind=DataKind.XGB_MODEL, data={"model_data": serialized_model})
            else:
                # initial run, starting from empty model
                dxo = model_learnable_to_dxo(model_learnable)
            return dxo.to_shareable()
        else:
            # return shareable saved from previous call to shareable_to_learnable
            return self.shareable

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Supporting TYPE == TYPE_XGB_MODEL

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.XGB_MODEL`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model

        dxo = from_shareable(shareable)

        if dxo.data_kind == DataKind.XGB_MODEL:
            model_update = dxo.data
            if not model_update:
                self.log_info(fl_ctx, "No model update found. Model will not be updated.")
            else:
                model_data_dict = model_update.get("model_data_dict")
                if model_data_dict:
                    # model update is from aggregator in bagging mode, update global model
                    model = base_model[ModelLearnableKey.WEIGHTS]
                    for update in model_data_dict:
                        model = update_model(model, update)
                    # remove model update dict from shareable that will be sent
                    dxo.data = {"model_data": model_update["model_data"]}
                else:
                    # model update is serialized full model currently in cyclic mode
                    model = model_update.get("model_data")
                base_model[ModelLearnableKey.WEIGHTS] = model
            self.shareable = dxo.to_shareable()
        else:
            raise ValueError("data_kind should be either DataKind.XGB_MODEL, but got {}".format(dxo.data_kind))
        return base_model
