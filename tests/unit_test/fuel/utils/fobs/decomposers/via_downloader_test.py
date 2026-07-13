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
from nvflare.fuel.f3.streaming.download_service import Downloadable, ProduceRC
from nvflare.fuel.f3.streaming.transfer_progress import DEFAULT_STREAMING_IDLE_TIMEOUT, STREAMING_IDLE_TIMEOUT
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers import via_downloader as via_downloader_module
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    RESULT_UPLOAD_PROGRESS_CTX_KEY,
    RESULT_UPLOAD_RECEIVER_IDS_CTX_KEY,
    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY,
    EncKey,
    EncType,
    ResultUploadProgressContextKey,
    ViaDownloaderDecomposer,
    clear_download_initiated,
    get_download_transactions,
)


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


class _FakeDownloadable(Downloadable):
    def __init__(self, script):
        super().__init__("base")
        self.script = list(script)
        self.set_transaction_calls = []

    def set_transaction(self, tx_id: str, ref_id: str):
        self.set_transaction_calls.append((tx_id, ref_id))

    def produce(self, state: dict, requester: str):
        return self.script.pop(0)


class _FakeObjectDownloader:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tx_id = "tx-1"
        self.added = []
        _FakeObjectDownloader.instances.append(self)

    def add_object(self, obj, ref_id=None):
        self.added.append((ref_id, obj))


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


def test_download_from_remote_cell_preserves_download_error(monkeypatch):
    decomposer = _DummyViaDownloader()
    error = "Declared blob size 2097152 exceeds configured limit 1048576"
    monkeypatch.setattr(decomposer, "download", lambda **_kwargs: (error, None))

    with pytest.raises(RuntimeError, match=error):
        decomposer._download_from_remote_cell(
            {fobs.FOBSContextKey.CELL: object()},
            {"fqcn": "server.job", "ref_id": "ref-1"},
        )


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

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_create_downloader_rejects_non_finite_generic_streaming_idle_timeout(self, monkeypatch, value):
        def fake_get_positive_float_var(var_name, default):
            if var_name == STREAMING_IDLE_TIMEOUT:
                return value
            return default

        monkeypatch.setattr(via_downloader_module.acu, "get_positive_float_var", fake_get_positive_float_var)

        decomposer = _DummyViaDownloader()
        with pytest.raises(ValueError, match="finite"):
            decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_create_downloader_rejects_non_finite_timeout_override(self, value):
        decomposer = _DummyViaDownloader()

        with pytest.raises(ValueError, match="finite"):
            decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()}, timeout_override=value)

    def test_create_downloader_floors_timeout_override_to_legacy_min_download_timeout(self, monkeypatch):
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
        decomposer._create_downloader({fobs.FOBSContextKey.CELL: object()}, timeout_override=600.0)

        assert observed["timeout"] == 1800.0

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


class TestResultUploadProgressWiring:
    def setup_method(self):
        clear_download_initiated()
        _FakeObjectDownloader.instances = []

    def _finalize(self, monkeypatch, fobs_ctx):
        monkeypatch.setattr(via_downloader_module, "ObjectDownloader", _FakeObjectDownloader)
        decomposer = _DummyViaDownloader()
        decomposer._finalize_download_tx(SimpleNamespace(fobs_ctx=fobs_ctx))
        return _FakeObjectDownloader.instances[0]

    def test_single_receiver_progress_installs_download_service_callback_and_captures_expected_pair(self, monkeypatch):
        events = []
        obj = _FakeDownloadable([])

        def fake_get_positive_float_var(var_name, default):
            if var_name == f"dummy_{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}":
                return 1.0
            return default

        monkeypatch.setattr(via_downloader_module.acu, "get_positive_float_var", fake_get_positive_float_var)
        fobs_ctx = {
            fobs.FOBSContextKey.CELL: object(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 1,
            fobs.FOBSContextKey.STREAM_PROGRESS_CB: lambda **kwargs: events.append(kwargs),
            RESULT_UPLOAD_PROGRESS_CTX_KEY: {ResultUploadProgressContextKey.STREAMING_IDLE_TIMEOUT: 7.0},
            via_downloader_module._CtxKey.MSG_ROOT_TTL: 1800.0,
            via_downloader_module._CtxKey.OBJECTS: [("ref-1", obj)],
        }

        downloader = self._finalize(monkeypatch, fobs_ctx)
        transactions = get_download_transactions()

        assert len(transactions) == 1
        assert transactions[0].tx_id == "tx-1"
        assert transactions[0].expected_pairs == (("ref-1", None),)
        assert downloader.kwargs["timeout"] == 7.0
        ref_id, added_obj = downloader.added[0]
        assert ref_id == "ref-1"
        assert added_obj is obj

        progress_cb = downloader.kwargs["progress_cb"]
        progress_cb(
            tx_id="tx-1",
            ref_id="ref-1",
            receiver_id="server",
            sequence=1,
            bytes_done=5,
            items_done=2,
            state="active",
        )
        progress_cb(
            tx_id="tx-1",
            ref_id="ref-1",
            receiver_id="server",
            sequence=2,
            bytes_done=5,
            items_done=2,
            state="completed",
        )

        assert events[0]["direction"] == "result_upload"
        assert events[0]["receiver_id"] is None
        assert events[0]["transfer_id"] == "ref-1"
        assert events[0]["bytes_done"] == 5
        assert events[0]["items_done"] == 2
        assert events[-1]["state"] == "completed"

    def test_multi_receiver_progress_captures_ref_receiver_pairs(self, monkeypatch):
        events = []
        created = []
        obj = _FakeDownloadable([])
        fobs_ctx = {
            fobs.FOBSContextKey.CELL: object(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 2,
            fobs.FOBSContextKey.STREAM_PROGRESS_CB: lambda **kwargs: events.append(kwargs),
            RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: lambda info: created.append(info),
            RESULT_UPLOAD_RECEIVER_IDS_CTX_KEY: ["server", "peer"],
            via_downloader_module._CtxKey.OBJECTS: [("ref-1", obj)],
        }

        downloader = self._finalize(monkeypatch, fobs_ctx)

        assert get_download_transactions()[0].expected_pairs == (("ref-1", "server"), ("ref-1", "peer"))
        assert created[0].tx_id == "tx-1"
        assert created[0].expected_pairs == (("ref-1", "server"), ("ref-1", "peer"))
        assert downloader.added == [("ref-1", obj)]
        downloader.kwargs["progress_cb"](
            tx_id="tx-1",
            ref_id="ref-1",
            receiver_id="server",
            sequence=1,
            bytes_done=3,
            items_done=None,
            state="active",
        )

        assert events[-1]["receiver_id"] == "server"

    def test_duplicate_receiver_ids_dedupe_expected_pairs_and_download_receiver_count(self, monkeypatch):
        obj = _FakeDownloadable([(ProduceRC.OK, b"abc", {})])
        events = []
        fobs_ctx = {
            fobs.FOBSContextKey.CELL: object(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 2,
            fobs.FOBSContextKey.STREAM_PROGRESS_CB: lambda **kwargs: events.append(kwargs),
            RESULT_UPLOAD_RECEIVER_IDS_CTX_KEY: ["server", "server"],
            via_downloader_module._CtxKey.MSG_ROOT_TTL: 1800.0,
            via_downloader_module._CtxKey.OBJECTS: [("ref-1", obj)],
        }

        downloader = self._finalize(monkeypatch, fobs_ctx)

        assert get_download_transactions()[0].expected_pairs == (("ref-1", "server"),)
        assert downloader.kwargs["num_receivers"] == 1
        assert downloader.kwargs["progress_cb"] is not None
        assert downloader.added == [("ref-1", obj)]

    def test_unknown_multi_receiver_transaction_is_not_marked_progress_trackable(self, monkeypatch):
        obj = _FakeDownloadable([(ProduceRC.OK, b"abc", {})])
        fobs_ctx = {
            fobs.FOBSContextKey.CELL: object(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 2,
            fobs.FOBSContextKey.STREAM_PROGRESS_CB: lambda **kwargs: None,
            RESULT_UPLOAD_PROGRESS_CTX_KEY: {ResultUploadProgressContextKey.STREAMING_IDLE_TIMEOUT: 7.0},
            via_downloader_module._CtxKey.MSG_ROOT_TTL: 1800.0,
            via_downloader_module._CtxKey.OBJECTS: [("ref-1", obj)],
        }

        downloader = self._finalize(monkeypatch, fobs_ctx)

        assert get_download_transactions() == ()
        assert downloader.added[0][1] is obj
        assert downloader.kwargs["timeout"] == 1800.0

    def test_finalize_download_tx_without_cell_returns_without_dereferencing_downloader(self):
        obj = _FakeDownloadable([(ProduceRC.OK, b"abc", {})])
        fobs_ctx = {
            via_downloader_module._CtxKey.OBJECTS: [("ref-1", obj)],
        }

        decomposer = _DummyViaDownloader()
        decomposer._finalize_download_tx(SimpleNamespace(fobs_ctx=fobs_ctx))

        assert get_download_transactions() == ()
        assert not getattr(via_downloader_module._tls, "download_initiated", False)


class TestForwardProgressCallback:
    def test_missing_message_metadata_is_not_replaced_with_ref_id(self):
        events = []
        progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(
            {fobs.FOBSContextKey.STREAM_PROGRESS_CB: lambda **kwargs: events.append(kwargs)},
            "ref-1",
        )

        progress_cb(sequence=1, bytes_done=0, state="active")

        assert events[0]["job_id"] is None
        assert events[0]["task_id"] is None
        assert events[0]["transfer_id"] == "ref-1"
        assert events[0]["direction"] == "task_payload_download"


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
