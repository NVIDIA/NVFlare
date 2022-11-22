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
from typing import Any

import numpy as np

from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.widgets.event_recorder import _CtxPropReq, _EventReq, _EventStats
from nvflare.fuel.utils import fobs


class TestCommonDecomposers:
    @classmethod
    def setup_class(cls):
        common_decomposers.register()

    def test_learnable(self):

        # Learnable is simply a dict with 2 extra methods
        learnable = Learnable()
        learnable["A"] = "foo"
        learnable["B"] = 123

        new_learnable = self._run_fobs(learnable)

        assert new_learnable["A"] == learnable["A"]
        assert new_learnable["B"] == learnable["B"]

    def test_model_learnable(self):

        # Learnable is simply a dict with 2 extra methods
        model_learnable = ModelLearnable()
        model_learnable["A"] = "bar"
        model_learnable["B"] = 456

        new_learnable = self._run_fobs(model_learnable)

        assert new_learnable["A"] == model_learnable["A"]
        assert new_learnable["B"] == model_learnable["B"]

    def test_np_float64(self):

        f64 = np.float64(1.234)

        new_f64 = self._run_fobs(f64)

        assert new_f64 == f64

    def test_np_array(self):

        npa = np.array([[1, 2, 3], [4, 5, 6]])

        new_npa = self._run_fobs(npa)

        assert (new_npa == npa).all()

    def test_ctx_prop_req(self):

        cpr = _CtxPropReq("data_type", True, False, True)

        new_cpr = self._run_fobs(cpr)

        assert new_cpr.dtype == cpr.dtype
        assert new_cpr.is_sticky == cpr.is_sticky
        assert new_cpr.is_private == cpr.is_private
        assert new_cpr.allow_none == cpr.allow_none

    def test_event_req(self):

        req = _EventReq(
            {"A": "foo"}, {"B": "bar"}, ["block_list1", "block_list2"], ["peer_block_list1", "peer_block_list2"]
        )

        new_req = self._run_fobs(req)

        assert new_req.ctx_reqs == req.ctx_reqs
        assert new_req.peer_ctx_reqs == req.peer_ctx_reqs
        assert new_req.ctx_block_list == req.ctx_block_list
        assert new_req.peer_ctx_block_list == req.peer_ctx_block_list

    def test_event_stats(self):

        stats = _EventStats()
        stats.call_count = 1
        stats.prop_missing = 2
        stats.prop_none_value = 3
        stats.prop_dtype_mismatch = 4
        stats.prop_attr_mismatch = 5
        stats.prop_block_list_violation = 6
        stats.peer_ctx_missing = 7

        new_stats = self._run_fobs(stats)

        assert new_stats.call_count == stats.call_count
        assert new_stats.prop_missing == stats.prop_missing
        assert new_stats.prop_none_value == stats.prop_none_value
        assert new_stats.prop_dtype_mismatch == stats.prop_dtype_mismatch
        assert new_stats.prop_attr_mismatch == stats.prop_attr_mismatch
        assert new_stats.prop_block_list_violation == stats.prop_block_list_violation
        assert new_stats.peer_ctx_missing == stats.peer_ctx_missing

    @staticmethod
    def _run_fobs(data: Any) -> Any:
        buf = fobs.dumps(data)
        return fobs.loads(buf)
