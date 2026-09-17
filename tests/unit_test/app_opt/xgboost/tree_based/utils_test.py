# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import unittest

import numpy as np
import xgboost as xgb

from nvflare.app_opt.xgboost.tree_based.utils import update_model


def _make_client_model(num_class: int, num_parallel_tree: int) -> dict:
    X = np.random.rand(100, 4)
    y = np.random.randint(0, num_class, 100)
    dtrain = xgb.DMatrix(X, label=y)
    objective = "multi:softprob" if num_class > 1 else "binary:logistic"
    params = {
        "objective": objective,
        "num_parallel_tree": num_parallel_tree,
        "tree_method": "hist",
    }
    if num_class > 1:
        params["num_class"] = num_class
    bst = xgb.train(params, dtrain, num_boost_round=1)
    return json.loads(bst.save_raw("json"))


class TestUpdateModel(unittest.TestCase):
    def test_binary_two_clients(self):
        """Binary: num_trees == num_parallel_tree, merging 2 clients is correct."""
        num_parallel_tree = 5
        client = _make_client_model(num_class=1, num_parallel_tree=num_parallel_tree)
        merged = update_model(copy.deepcopy(client), copy.deepcopy(client))
        model_body = merged["learner"]["gradient_booster"]["model"]
        expected = 2 * num_parallel_tree
        self.assertEqual(len(model_body["trees"]), expected)
        self.assertEqual(int(model_body["gbtree_model_param"]["num_trees"]), expected)

    def test_multiclass_two_clients(self):
        """multi:softprob: each client has num_class * num_parallel_tree trees.
        Merging 2 clients must produce 2 * num_class * num_parallel_tree trees."""
        num_class, num_parallel_tree = 6, 5
        client = _make_client_model(num_class=num_class, num_parallel_tree=num_parallel_tree)
        trees_per_client = len(client["learner"]["gradient_booster"]["model"]["trees"])
        client_tree_info = client["learner"]["gradient_booster"]["model"]["tree_info"]
        self.assertEqual(trees_per_client, num_class * num_parallel_tree)

        merged = update_model(copy.deepcopy(client), copy.deepcopy(client))
        model_body = merged["learner"]["gradient_booster"]["model"]
        expected = 2 * trees_per_client
        self.assertEqual(
            len(model_body["trees"]),
            expected,
            f"Expected {expected} trees, got {len(model_body['trees'])} " f"(num_class blindness bug)",
        )
        self.assertEqual(int(model_body["gbtree_model_param"]["num_trees"]), expected)
        indptr = model_body["iteration_indptr"]
        self.assertEqual(indptr[-1], expected)
        self.assertEqual(model_body["tree_info"], client_tree_info + client_tree_info)

    def test_tree_ids_are_sequential(self):
        """After merging, tree ids must be 0..N-1 with no gaps."""
        client = _make_client_model(num_class=6, num_parallel_tree=5)
        merged = update_model(copy.deepcopy(client), copy.deepcopy(client))
        trees = merged["learner"]["gradient_booster"]["model"]["trees"]
        ids = [t["id"] for t in trees]
        self.assertEqual(ids, list(range(len(trees))))

    def test_first_client_none_prev(self):
        """update_model with no prior model returns the client model unchanged."""
        client = _make_client_model(num_class=6, num_parallel_tree=5)
        result = update_model(None, copy.deepcopy(client))
        n = len(result["learner"]["gradient_booster"]["model"]["trees"])
        self.assertEqual(n, 30)

    def test_iteration_indptr_multiclass(self):
        """iteration_indptr must advance by add_num_trees (not num_parallel_tree)."""
        num_class, num_parallel_tree = 6, 5
        client = _make_client_model(num_class=num_class, num_parallel_tree=num_parallel_tree)
        trees_per_client = num_class * num_parallel_tree
        merged = update_model(copy.deepcopy(client), copy.deepcopy(client))
        indptr = merged["learner"]["gradient_booster"]["model"]["iteration_indptr"]
        self.assertEqual(indptr, [0, trees_per_client, 2 * trees_per_client])


if __name__ == "__main__":
    unittest.main()
