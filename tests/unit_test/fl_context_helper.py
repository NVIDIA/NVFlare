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

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext


def make_fl_context(engine=None, identity_name=None, run_num=None) -> FLContext:
    """Build an FLContext preloaded with the props commonly needed by unit tests."""
    fl_ctx = FLContext()
    if engine is not None:
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    if identity_name is not None:
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, identity_name, private=False, sticky=False)
    if run_num is not None:
        fl_ctx.set_prop(ReservedKey.RUN_NUM, run_num, private=True, sticky=False)
    return fl_ctx
