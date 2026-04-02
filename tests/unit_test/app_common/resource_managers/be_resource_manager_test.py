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

import uuid

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.app_common.resource_managers.be_resource_manager import BEResourceManager


def _fl_ctx():
    return FLContext()


class TestBEResourceManagerIsSpec:
    def test_is_subclass_of_resource_manager_spec(self):
        assert issubclass(BEResourceManager, ResourceManagerSpec)

    def test_instance_is_resource_manager_spec(self):
        assert isinstance(BEResourceManager(), ResourceManagerSpec)


class TestBEResourceManagerCheckResources:
    def test_returns_true(self):
        ok, _ = BEResourceManager().check_resources({"num_of_gpus": 1}, _fl_ctx())
        assert ok is True

    def test_returns_string_token(self):
        _, token = BEResourceManager().check_resources({}, _fl_ctx())
        assert isinstance(token, str)

    def test_token_is_valid_uuid(self):
        _, token = BEResourceManager().check_resources({}, _fl_ctx())
        uuid.UUID(token)  # raises if invalid

    def test_tokens_are_unique_per_call(self):
        mgr = BEResourceManager()
        _, t1 = mgr.check_resources({}, _fl_ctx())
        _, t2 = mgr.check_resources({}, _fl_ctx())
        assert t1 != t2

    def test_empty_requirement_approved(self):
        ok, _ = BEResourceManager().check_resources({}, _fl_ctx())
        assert ok is True

    def test_non_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            BEResourceManager().check_resources("not-a-dict", _fl_ctx())

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            BEResourceManager().check_resources([1, 2], _fl_ctx())

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            BEResourceManager().check_resources(None, _fl_ctx())


class TestBEResourceManagerCancelResources:
    def test_returns_none(self):
        result = BEResourceManager().cancel_resources({"num_of_gpus": 1}, "tok", _fl_ctx())
        assert result is None

    def test_does_not_raise(self):
        BEResourceManager().cancel_resources({}, "tok", _fl_ctx())


class TestBEResourceManagerAllocateResources:
    def test_returns_empty_dict(self):
        result = BEResourceManager().allocate_resources({"num_of_gpus": 2}, "tok", _fl_ctx())
        assert result == {}

    def test_returns_dict_type(self):
        result = BEResourceManager().allocate_resources({}, "tok", _fl_ctx())
        assert isinstance(result, dict)


class TestBEResourceManagerFreeResources:
    def test_returns_none(self):
        result = BEResourceManager().free_resources({"num_of_gpus": 1}, "tok", _fl_ctx())
        assert result is None

    def test_does_not_raise(self):
        BEResourceManager().free_resources({}, "tok", _fl_ctx())


class TestBEResourceManagerReportResources:
    def test_returns_empty_dict(self):
        assert BEResourceManager().report_resources(_fl_ctx()) == {}

    def test_returns_dict_type(self):
        assert isinstance(BEResourceManager().report_resources(_fl_ctx()), dict)


class TestBEResourceManagerLifecycle:
    def test_check_allocate_free_cycle(self):
        mgr = BEResourceManager()
        ctx = _fl_ctx()
        req = {"num_of_gpus": 4}

        ok, token = mgr.check_resources(req, ctx)
        assert ok is True

        allocated = mgr.allocate_resources(req, token, ctx)
        assert isinstance(allocated, dict)

        mgr.free_resources(allocated, token, ctx)

    def test_check_cancel_cycle(self):
        mgr = BEResourceManager()
        ctx = _fl_ctx()
        ok, token = mgr.check_resources({"num_of_gpus": 2}, ctx)
        assert ok is True
        mgr.cancel_resources({"num_of_gpus": 2}, token, ctx)
