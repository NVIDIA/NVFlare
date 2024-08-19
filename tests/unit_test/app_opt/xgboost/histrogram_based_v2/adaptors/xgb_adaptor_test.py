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

from unittest.mock import patch

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBClientAdaptor, XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant


@patch.multiple(XGBServerAdaptor, __abstractmethods__=set())
class TestXGBServerAdaptor:
    def test_configure(self):
        xgb_adaptor = XGBServerAdaptor(True)
        config = {Constant.CONF_KEY_WORLD_SIZE: 66}
        ctx = FLContext()
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 66


@patch.multiple(XGBClientAdaptor, __abstractmethods__=set())
class TestXGBClientAdaptor:
    def test_configure(self):
        xgb_adaptor = XGBClientAdaptor(True, 1, 10)
        config = {
            Constant.CONF_KEY_CLIENT_RANKS: {"site-test": 1},
            Constant.CONF_KEY_NUM_ROUNDS: 100,
            Constant.CONF_KEY_DATA_SPLIT_MODE: 0,
            Constant.CONF_KEY_SECURE_TRAINING: False,
            Constant.CONF_KEY_XGB_PARAMS: {"depth": 1},
            Constant.CONF_KEY_DISABLE_VERSION_CHECK: False,
        }
        ctx = FLContext()
        ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-test")
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 1
        assert xgb_adaptor.rank == 1
        assert xgb_adaptor.num_rounds == 100
