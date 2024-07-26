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

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.flower.executor import FlowerExecutor
from nvflare.app_opt.flower.mock.applet import MockClientApplet, MockClientPyApplet


class MockExecutor(FlowerExecutor):
    def __init__(self):
        FlowerExecutor.__init__(self)

    def get_applet(self, fl_ctx: FLContext):
        return MockClientApplet()


class MockPyExecutor(FlowerExecutor):
    def __init__(self, in_process=True):
        FlowerExecutor.__init__(self)
        self.in_process = in_process

    def get_applet(self, fl_ctx: FLContext):
        return MockClientPyApplet(self.in_process)
