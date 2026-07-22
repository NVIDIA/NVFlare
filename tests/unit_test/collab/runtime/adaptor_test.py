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

from unittest.mock import MagicMock

from nvflare.apis.fl_context import FLContext
from nvflare.collab import collab
from nvflare.collab.api.app import App
from nvflare.collab.api.constants import FL_CONTEXT_PROP
from nvflare.collab.runtime.flare.adaptor import CollabAdaptor


def test_process_config_with_default_resource_dirs():
    adaptor = CollabAdaptor()
    app = App(object(), "app")
    fl_ctx = MagicMock(spec=FLContext)

    assert adaptor.process_config(app, fl_ctx) is None
    assert app.get_resource_dirs() == {}
    assert app.get_prop(FL_CONTEXT_PROP) is fl_ctx

    with app.new_context(caller="app", callee="app"):
        assert collab.fl_ctx is fl_ctx
