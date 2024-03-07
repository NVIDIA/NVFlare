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

from unittest.mock import patch, Mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor import XGBServerAdaptor, XGBAdaptor, XGBClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runner import XGBRunner
from nvflare.app_opt.xgboost.histogram_based_v2.sender import Sender
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant


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


class TestXGBServerAdaptor:
    @patch.multiple(XGBServerAdaptor, __abstractmethods__=set())
    def test_configure(self):
        xgb_adaptor = XGBServerAdaptor()
        config = {
            Constant.CONF_KEY_WORLD_SIZE: 66
        }
        ctx = FLContext()
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 66


@patch.multiple(XGBClientAdaptor, __abstractmethods__=set())
class TestXGBClientAdaptor:
    def test_configure(self):
        xgb_adaptor = XGBClientAdaptor()
        config = {
            Constant.CONF_KEY_WORLD_SIZE: 66,
            Constant.CONF_KEY_RANK: 44,
            Constant.CONF_KEY_NUM_ROUNDS: 100
        }
        ctx = FLContext()
        xgb_adaptor.configure(config, ctx)
        assert xgb_adaptor.world_size == 66
        assert xgb_adaptor.rank == 44
        assert xgb_adaptor.num_rounds == 100

    def test_send(self):
        xgb_adaptor = XGBClientAdaptor()
        sender = Mock(spec=Sender)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = b"hello"
        sender.send_to_server.return_value = reply
        abort_signal = Signal()
        xgb_adaptor.set_abort_signal(abort_signal)
        xgb_adaptor.set_sender(sender)
        assert xgb_adaptor.sender == sender
        assert xgb_adaptor._send_request("", Shareable()) == b"hello"
