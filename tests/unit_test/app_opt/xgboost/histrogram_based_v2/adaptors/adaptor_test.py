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

from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.adaptor import AppAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner


@patch.multiple(AppAdaptor, __abstractmethods__=set())
class TestAppAdaptor:
    def test_set_abort_signal(self):
        app_adaptor = AppAdaptor("_test", True)
        abort_signal = Signal()
        app_adaptor.set_abort_signal(abort_signal)
        abort_signal.trigger("cool")
        assert app_adaptor.abort_signal.triggered

    @patch.multiple(AppRunner, __abstractmethods__=set())
    def test_set_runner(self):
        runner = AppRunner()
        app_adaptor = AppAdaptor("_test", True)

        app_adaptor.set_runner(runner)

        assert app_adaptor.app_runner == runner
