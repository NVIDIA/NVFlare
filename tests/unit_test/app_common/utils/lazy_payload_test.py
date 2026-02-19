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

from nvflare.app_common.utils.lazy_payload import cleanup_inplace, contains_lazy, resolve_inplace


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


class _GuardedLazyRef(_LazyRef):
    def resolve(self):
        if self._temp_ref.cleaned:
            raise RuntimeError("temp ref cleaned before all lazy refs were resolved")
        return super().resolve()


class _Cleaner:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class TestLazyPayloadUtils:
    def test_contains_lazy(self):
        temp_ref = _TempRef()
        payload = {"a": 1, "b": [_LazyRef(2, temp_ref)]}
        assert contains_lazy(payload) is True
        assert contains_lazy({"a": 1, "b": [2, 3]}) is False

    def test_resolve_inplace(self):
        temp_ref = _TempRef()
        lazy_ref = _LazyRef(9, temp_ref)
        payload = {"a": lazy_ref, "b": [1, _LazyRef(7, temp_ref)]}
        resolved = resolve_inplace(payload)
        assert resolved is payload
        assert payload["a"] == 9
        assert payload["b"][1] == 7
        assert lazy_ref.resolve_calls == 1

    def test_resolve_inplace_with_cleanup(self):
        temp_ref = _TempRef()
        payload = {"a": _LazyRef(9, temp_ref)}
        resolve_inplace(payload, cleanup_resolved=True)
        assert payload["a"] == 9
        assert temp_ref.cleaned is True

    def test_resolve_inplace_with_cleanup_multiple_refs(self):
        temp_ref = _TempRef()
        payload = {"a": _GuardedLazyRef(9, temp_ref), "b": _GuardedLazyRef(7, temp_ref)}
        resolve_inplace(payload, cleanup_resolved=True)
        assert payload["a"] == 9
        assert payload["b"] == 7
        assert temp_ref.cleaned is True

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
