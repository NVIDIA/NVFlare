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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.event import fire_event_to_components


def test_same_class_handler_diagnostics_preserve_both_failures_in_order():
    errors = [ValueError("first failure"), ValueError("second failure")]

    class FailingComponent(FLComponent):
        def __init__(self, error):
            super().__init__()
            self.error = error

        def handle_event(self, event_type, fl_ctx):
            raise self.error

    components = [FailingComponent(error) for error in errors]
    ctx = FLContext()
    captured = []
    fire_event_to_components("test", components, ctx, exception_handler=captured.append)
    diagnostics = ctx.get_prop(FLContextKey.EXCEPTIONS)
    assert captured == errors
    assert list(diagnostics.values()) == errors
    assert diagnostics[components[0].name] is errors[0]
