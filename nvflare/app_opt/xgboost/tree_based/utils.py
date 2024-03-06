# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


def _get_xgboost_model_attr(xgb_model):
    num_parallel_tree = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"]
    )
    num_trees = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    return num_parallel_tree, num_trees


def update_model(prev_model, model_update):
    if not prev_model:
        return model_update
    else:
        # Append all trees
        # get the parameters
        pre_num_parallel_tree, pre_num_trees = _get_xgboost_model_attr(prev_model)
        cur_num_parallel_tree, add_num_trees = _get_xgboost_model_attr(model_update)

        # check num_parallel_tree, should be consistent
        if cur_num_parallel_tree != pre_num_parallel_tree:
            raise ValueError(
                f"add_num_parallel_tree should not change, previous {pre_num_parallel_tree}, current {cur_num_parallel_tree}"
            )
        prev_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
            pre_num_trees + cur_num_parallel_tree
        )
        # append the new trees
        append_info = model_update["learner"]["gradient_booster"]["model"]["trees"]
        for tree_ct in range(cur_num_parallel_tree):
            append_info[tree_ct]["id"] = pre_num_trees + tree_ct
            prev_model["learner"]["gradient_booster"]["model"]["trees"].append(append_info[tree_ct])
            prev_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
        # append iteration_indptr
        prev_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
            pre_num_trees + cur_num_parallel_tree
        )
        return prev_model
