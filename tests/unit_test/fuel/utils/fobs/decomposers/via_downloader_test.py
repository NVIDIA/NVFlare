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

from types import SimpleNamespace

from nvflare.fuel.utils.fobs.decomposers.via_downloader import EncKey, EncType, ViaDownloaderDecomposer


class _DummyViaDownloader(ViaDownloaderDecomposer):
    def __init__(self):
        super().__init__(max_chunk_size=1, config_var_prefix="dummy_")

    def supported_type(self):
        return object

    def to_downloadable(self, items: dict, max_chunk_size: int, fobs_ctx: dict):
        raise NotImplementedError

    def download(
        self,
        from_fqcn: str,
        ref_id: str,
        per_request_timeout: float,
        cell,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def get_download_dot(self) -> int:
        return 9999

    def native_decompose(self, target, manager=None) -> bytes:
        return b""

    def native_recompose(self, data: bytes, manager=None):
        return data


class _ItemsWithNonCallableLazyRef:
    make_lazy_ref = "not-callable"

    def __contains__(self, item_id):
        return item_id == "T0"

    def get(self, item_id):
        if item_id == "T0":
            return "from_get"
        return None


class _ItemsWithCallableLazyRef:
    def __contains__(self, item_id):
        return item_id == "T0"

    def make_lazy_ref(self, item_id):
        return f"lazy_{item_id}"

    def get(self, item_id):
        return f"get_{item_id}"


class TestViaDownloaderRecomposeLazyRefGuard:
    def test_non_callable_make_lazy_ref_falls_back_to_get(self):
        decomposer = _DummyViaDownloader()
        items = _ItemsWithNonCallableLazyRef()
        manager = SimpleNamespace(fobs_ctx={decomposer.items_key: items})

        result = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: "T0"}, manager)
        assert result == "from_get"

    def test_callable_make_lazy_ref_is_used(self):
        decomposer = _DummyViaDownloader()
        items = _ItemsWithCallableLazyRef()
        manager = SimpleNamespace(fobs_ctx={decomposer.items_key: items})

        result = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: "T0"}, manager)
        assert result == "lazy_T0"
