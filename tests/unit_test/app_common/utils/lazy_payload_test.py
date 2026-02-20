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

from nvflare.app_common.utils.lazy_payload import cleanup_inplace


class _TempRef:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class _LazyRef:
    def __init__(self, value, temp_ref: _TempRef):
        self.value = value
        self._temp_ref = temp_ref
        self.resolve_calls = 0

    def resolve(self):
        self.resolve_calls += 1
        return self.value


class _Cleaner:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class TestLazyPayloadUtils:
    def test_cleanup_inplace_with_lazy_refs(self):
        temp_ref = _TempRef()
        payload = {"a": _LazyRef(1, temp_ref), "b": _LazyRef(2, temp_ref)}
        cleaned = cleanup_inplace(payload)
        assert cleaned is True
        assert temp_ref.cleaned is True

    def test_cleanup_inplace_with_cleanup_object(self):
        cleaner = _Cleaner()
        cleaned = cleanup_inplace({"x": cleaner})
        assert cleaned is True
        assert cleaner.cleaned is True
