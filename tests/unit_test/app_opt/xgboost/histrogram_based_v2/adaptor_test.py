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

from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor import XGBAdaptor, XGBClientAdaptor, XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runner import XGBRunner


@patch.multiple(XGBAdaptor, __abstractmethods__=set())
class TestXGBAdaptor:
    def test_set_abort_signal(self):
        xgb_adaptor = XGBAdaptor()
        abort_signal = Signal()
        xgb_adaptor.set_abort_signal(abort_signal)
        abort_signal.trigger("cool")
        assert xgb_adaptor.abort_signal.triggered

    @patch.multiple(XGBRunner, __abstractmethods__=set())
    def test_set_runner(self):
        runner = XGBRunner()
        xgb_adaptor = XGBAdaptor()

        xgb_adaptor.set_runner(runner)

        assert xgb_adaptor.xgb_runner == runner


class MockEngine:
    def __init__(self, run_name="adaptor_test"):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            job_id=run_name,
            public_stickers={},
            private_stickers={},
        )

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass


class TestXGBServerAdaptor:
    @patch.multiple(XGBServerAdaptor, __abstractmethods__=set())
    def test_configure(self):
        xgb_adaptor = XGBServerAdaptor()
        config = {Constant.CONF_KEY_WORLD_SIZE: 66}
        ctx = FLContext()
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 66


@patch.multiple(XGBClientAdaptor, __abstractmethods__=set())
class TestXGBClientAdaptor:
    def test_configure(self):
        xgb_adaptor = XGBClientAdaptor(10, 100)
        config = {Constant.CONF_KEY_WORLD_SIZE: 66, Constant.CONF_KEY_RANK: 44, Constant.CONF_KEY_NUM_ROUNDS: 100}
        ctx = MockEngine().new_context()
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 66
        assert xgb_adaptor.rank == 44
        assert xgb_adaptor.num_rounds == 100

    def test_send(self):
        pass
        # xgb_adaptor = XGBClientAdaptor(10, 100)
        # ctx = MockEngine().new_context()
        # config = {Constant.CONF_KEY_WORLD_SIZE: 66, Constant.CONF_KEY_RANK: 44, Constant.CONF_KEY_NUM_ROUNDS: 100}
        # xgb_adaptor.configure(config, ctx)
        # reply = Shareable()
        # reply.set_header(Constant.MSG_KEY_XGB_OP, "")
        # reply[Constant.PARAM_KEY_RCV_BUF] = b"hello"
        # # xgb_adaptor._send_request.return_value = reply
        # abort_signal = Signal()
        # xgb_adaptor.set_abort_signal(abort_signal)
        # assert xgb_adaptor._send_request("", Shareable()) == b"hello"
