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

import pytest

from nvflare.apis.fl_constant import ConfigVarName
from nvflare.fuel.f3.streaming.transfer_progress import DEFAULT_STREAMING_IDLE_TIMEOUT, STREAMING_IDLE_TIMEOUT
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers import via_downloader as via_downloader_module
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
        progress_cb=None,
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

    def test_missing_downloaded_items_raises(self):
        decomposer = _DummyViaDownloader()
        manager = SimpleNamespace(fobs_ctx={})

        with pytest.raises(RuntimeError, match="FOBS download data is missing"):
            decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: "T0"}, manager)

    def test_missing_downloaded_item_raises(self):
        decomposer = _DummyViaDownloader()
        manager = SimpleNamespace(fobs_ctx={decomposer.items_key: {}})

        with pytest.raises(RuntimeError, match="FOBS download data is incomplete"):
            decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: "T0"}, manager)

    def test_none_downloaded_item_raises(self):
        decomposer = _DummyViaDownloader()
        manager = SimpleNamespace(fobs_ctx={decomposer.items_key: {"T0": None}})

        with pytest.raises(RuntimeError, match="item T0 is None"):
            decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: "T0"}, manager)


class TestViaDownloaderTimeoutPolicy:
    def test_create_downloader_uses_generic_default_when_no_timeouts_are_configured(self, monkeypatch):
        observed = {}

        def fake_get_positive_float_var(var_name, default):
            return default

        class FakeObjectDownloader:
            def __init__(self, **kwargs):
                observed.update(kwargs)

        monkeypatch.setattr(via_downloader_module.acu, "get_positive_float_var", fake_get_positive_float_var)
        monkeypatch.setattr(via_downloader_module, "ObjectDownloader", FakeObjectDownloader)

        decomposer = _DummyViaDownloader()
        decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()})

        assert observed["timeout"] == DEFAULT_STREAMING_IDLE_TIMEOUT

    def test_create_downloader_uses_generic_streaming_idle_timeout_as_default_floor(self, monkeypatch):
        observed = {}

        def fake_get_positive_float_var(var_name, default):
            if var_name == STREAMING_IDLE_TIMEOUT:
                return 1200.0
            if var_name == f"dummy_{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}":
                return default
            return default

        class FakeObjectDownloader:
            def __init__(self, **kwargs):
                observed.update(kwargs)

        monkeypatch.setattr(via_downloader_module.acu, "get_positive_float_var", fake_get_positive_float_var)
        monkeypatch.setattr(via_downloader_module, "ObjectDownloader", FakeObjectDownloader)

        decomposer = _DummyViaDownloader()
        decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()})

        assert observed["timeout"] == 1200.0

    def test_create_downloader_honors_explicit_legacy_min_download_timeout(self, monkeypatch):
        observed = {}

        def fake_get_positive_float_var(var_name, default):
            if var_name == STREAMING_IDLE_TIMEOUT:
                return 1200.0
            if var_name == f"dummy_{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}":
                return 1800.0
            return default

        class FakeObjectDownloader:
            def __init__(self, **kwargs):
                observed.update(kwargs)

        monkeypatch.setattr(via_downloader_module.acu, "get_positive_float_var", fake_get_positive_float_var)
        monkeypatch.setattr(via_downloader_module, "ObjectDownloader", FakeObjectDownloader)

        decomposer = _DummyViaDownloader()
        decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()})

        assert observed["timeout"] == 1800.0


class TestConcreteViaDownloaderProgressCallback:
    def test_numpy_decomposer_passes_progress_callback_to_download_object(self, monkeypatch):
        from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
        from nvflare.app_common.np import np_downloader

        observed = {}
        progress_cb = object()

        def fake_download_object(**kwargs):
            observed.update(kwargs)
            kwargs["consumer"].download_completed(kwargs["ref_id"])

        monkeypatch.setattr(np_downloader, "download_object", fake_download_object)

        err, result = NumpyArrayDecomposer().download(
            from_fqcn="server",
            ref_id="ref-1",
            per_request_timeout=1.0,
            cell=object(),
            progress_cb=progress_cb,
        )

        assert err is None
        assert result is None
        assert observed["progress_cb"] is progress_cb

    def test_tensor_decomposer_passes_progress_callback_to_download_object(self, monkeypatch):
        pytest.importorskip("torch")
        from nvflare.app_opt.pt import tensor_downloader
        from nvflare.app_opt.pt.decomposers import TensorDecomposer

        observed = {}
        progress_cb = object()

        class FakeCell:
            def get_fobs_context(self):
                return {}

        def fake_download_object(**kwargs):
            observed.update(kwargs)
            kwargs["consumer"].download_completed(kwargs["ref_id"])

        monkeypatch.setattr(tensor_downloader, "download_object", fake_download_object)

        err, result = TensorDecomposer().download(
            from_fqcn="server",
            ref_id="ref-1",
            per_request_timeout=1.0,
            cell=FakeCell(),
            progress_cb=progress_cb,
        )

        assert err is None
        assert result is None
        assert observed["progress_cb"] is progress_cb

    def test_tensor_disk_decomposer_passes_progress_callback_to_download_object(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        from nvflare.app_opt.pt import tensor_downloader
        from nvflare.app_opt.pt.decomposers import TensorDecomposer

        observed = {}
        progress_cb = object()

        class FakeCell:
            def get_fobs_context(self):
                return {
                    "enable_tensor_disk_offload": True,
                    tensor_downloader._TENSOR_DISK_OFFLOAD_ROOT_DIR: str(tmp_path),
                }

        def fake_download_object(**kwargs):
            observed.update(kwargs)
            kwargs["consumer"].result = {}
            kwargs["consumer"].download_completed(kwargs["ref_id"])

        monkeypatch.setattr(tensor_downloader, "download_object", fake_download_object)

        err, result = TensorDecomposer().download(
            from_fqcn="server",
            ref_id="ref-1",
            per_request_timeout=1.0,
            cell=FakeCell(),
            progress_cb=progress_cb,
        )

        assert err is None
        assert result is not None
        assert observed["progress_cb"] is progress_cb
        result.cleanup()
