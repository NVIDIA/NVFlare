# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_context import FLContext, FLContextManager


class TestFLContext:
    def test_add_item(self):

        expected = {"x": 1, "y": 2}

        fl_ctx = FLContext()

        fl_ctx.set_prop("x", 1, private=False)
        fl_ctx.set_prop("y", 2, private=False)

        assert fl_ctx.get_prop("y") == 2

        assert fl_ctx.get_all_public_props() == expected

    def test_add_with_private_item(self):

        expected = {"x": 1, "y": 2}

        fl_ctx = FLContext()

        fl_ctx.set_prop("x", 1, private=False)
        fl_ctx.set_prop("y", 2, private=False)
        fl_ctx.set_prop("z", 3, private=True)

        assert fl_ctx.get_prop("y") == 2

        assert fl_ctx.get_all_public_props() == expected

    def test_set_items(self):

        expected = {"_public_x": 1, "_public_y": 2}

        fl_ctx = FLContext()

        fl_ctx.set_prop("z", 3, private=False)

        # Overwrite the existing public_props.
        fl_ctx.set_public_props(expected)

        assert fl_ctx.get_all_public_props() == expected

    def test_not_allow_duplicate_key(self):
        fl_ctx = FLContext()

        fl_ctx.set_prop("x", 1, private=False)
        fl_ctx.set_prop("y", 2, private=False)
        fl_ctx.set_prop("z", 3, private=True)

        assert fl_ctx.set_prop("y", 20, private=False)
        assert fl_ctx.get_prop("y") == 20
        assert not fl_ctx.set_prop("y", 4, private=True)

    def test_remove_prop(self):
        fl_ctx = FLContext()

        assert fl_ctx.set_prop("x", 1, private=False)
        assert fl_ctx.set_prop("y", 2, private=False)

        fl_ctx.remove_prop("y")
        assert fl_ctx.set_prop("y", 20, private=True)

        assert fl_ctx.get_prop("y") == 20

    def test_sticky_prop(self):
        mgr = FLContextManager()
        ctx1 = mgr.new_context()
        ctx2 = mgr.new_context()
        ctx1.set_prop(key="x", value=1, private=True, sticky=True)
        assert ctx2.get_prop("x") == 1
        ctx2.set_prop(key="x", value=2, private=True, sticky=True)
        assert ctx2.get_prop("x") == 2
        assert ctx1.get_prop("x") == 2
