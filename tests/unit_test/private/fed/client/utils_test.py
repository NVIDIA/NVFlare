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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client.utils import determine_parent_fqcn


def test_determine_parent_fqcn_uses_task_routing_override():
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.TASK_ROUTING_TARGET, "custom.target", private=True, sticky=True)

    assert determine_parent_fqcn({"fqsn": "org/site/child"}, fl_ctx) == "custom.target"
