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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save

from nvflare.apis.fl_constant import ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.tensor_disk_offload_context import (
    _TENSOR_DISK_OFFLOAD_CONTEXT,
    _TENSOR_DISK_OFFLOAD_ROOT_DIR,
)
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.tensor_downloader import DiskTensorConsumer
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import CellChannel, Encoding, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.streaming.download_service import ProduceRC
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import FOBSContextKey, dots
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef, LazyDownloadRefDecomposer


@pytest.mark.parametrize("pipelining", [False, True])
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "finalize_mid",
        "finalize_eof",
        "fedavg_mid",
        "fedavg_before_start",
        "malformed",
        "disk_full",
        "source_error",
        "network_error",
        "finalize_network_error",
    ],
)
def test_streamed_update_finalization(tmp_path, monkeypatch, outcome, pipelining):
    """Use real FOBS decode and download loop; substitute only network replies and exit."""
    fobs.register(TensorDecomposer)
    fobs.register(LazyDownloadRefDecomposer)
    cell = MagicMock()
    cell.decode_pass_through_channels = set()
    cell.decode_pass_through_topics = set()
    context = {
        FOBSContextKey.CELL: cell,
        FOBSContextKey.TENSOR_DISK_OFFLOAD: True,
        _TENSOR_DISK_OFFLOAD_ROOT_DIR: str(tmp_path),
        _TENSOR_DISK_OFFLOAD_CONTEXT: None,
    }
    cell.get_fobs_context.side_effect = lambda props=None: {**context, **(props or {})}
    cell.update_fobs_context.side_effect = context.update
    tensors = {"T0": torch.tensor([1.0]), "T1": torch.tensor([2.0])}
    chunks = [save({key: value}) for key, value in tensors.items()]
    if outcome == "malformed":
        chunks[1] = b"invalid"
    replies = [
        make_reply(ReturnCode.OK, body={"status": ProduceRC.OK, "state": {"index": i + 1}, "data": [chunk]})
        for i, chunk in enumerate(chunks)
    ]
    replies.append(make_reply(ReturnCode.OK, body={"status": ProduceRC.EOF}))
    if outcome == "source_error":
        replies[1] = make_reply(ReturnCode.OK, body={"status": ProduceRC.ERROR})
    elif outcome in {"network_error", "finalize_network_error"}:
        replies[1] = OSError("connection lost")
    cell.send_request.side_effect = replies
    comm = WFCommServer()
    consume = DiskTensorConsumer.consume_items
    consumed = 0
    first_chunk, resume_download = Event(), Event()

    if outcome == "fedavg_before_start":
        download = TensorDecomposer.download

        def delayed_download(self, *args, **kwargs):
            # Adapter already captured the root's enabled decode context.
            first_chunk.set()
            assert resume_download.wait(5), "FedAvg cleanup did not finish"
            return download(self, *args, **kwargs)

        monkeypatch.setattr(TensorDecomposer, "download", delayed_download)

    def consume_items(consumer, items, result):
        nonlocal consumed
        if outcome == "disk_full" and consumed:
            raise OSError("No space left on device")
        result = consume(consumer, items, result)
        consumed += 1
        if outcome in {"finalize_mid", "finalize_network_error"} and consumed == 1:
            comm.finalize_run(FLContext())
        elif outcome == "fedavg_mid" and consumed == 1:
            first_chunk.set()
            assert resume_download.wait(5), "FedAvg cleanup did not finish"
        return result

    monkeypatch.setattr(DiskTensorConsumer, "supports_pipelining", pipelining)
    monkeypatch.setattr(DiskTensorConsumer, "consume_items", consume_items)
    if outcome == "finalize_eof":
        monkeypatch.setattr(DiskTensorConsumer, "download_completed", lambda *args: comm.finalize_run(FLContext()))

    future = MagicMock()
    future.headers = {
        StreamHeaderKey.CHANNEL: CellChannel.SERVER_COMMAND,
        StreamHeaderKey.TOPIC: ServerCommandNames.SUBMIT_UPDATE,
        StreamHeaderKey.STREAM_REQ_ID: "stream-1",
        StreamHeaderKey.PAYLOAD_ENCODING: Encoding.FOBS,
        MessageHeaderKey.REQ_ID: "request-1",
        MessageHeaderKey.ORIGIN: "site-1.job",
    }
    future.result.return_value = fobs.dumps(
        {key: LazyDownloadRef("site-1.job", "result-ref", key, dots.TENSOR_DOWNLOAD) for key in tensors}
    )
    callback = MagicMock(return_value=make_reply(ReturnCode.OK))
    adapter = Adapter(callback, SimpleNamespace(fqcn="server.job"), cell)
    fails = outcome in {"malformed", "disk_full", "source_error", "network_error"}
    with patch("nvflare.fuel.f3.cellnet.cell.os._exit", side_effect=SystemExit(1)) as exit_process:
        if fails:
            with pytest.raises(SystemExit):
                adapter.call(future)
            exit_process.assert_called_once_with(1)
        elif outcome in {"fedavg_mid", "fedavg_before_start"}:
            controller = FedAvg(num_clients=2, num_rounds=1, model=tensors, enable_tensor_disk_offload=True)
            controller.fl_ctx = FLContext()
            controller.engine = SimpleNamespace(get_cell=lambda: cell)
            controller.sample_clients = lambda _: ["good", "retired"]
            controller.get_num_standing_tasks = lambda: 0  # The slow task retired before decode finished.
            controller.event = MagicMock()
            controller.fire_event_with_data = MagicMock()
            controller.save_model = MagicMock()
            previous_context = dict(context)
            other_dir = tmp_path / "other-workflow"
            other_dir.mkdir()
            other = DiskTensorConsumer(str(other_dir))
            pending = []
            roots = []
            with ThreadPoolExecutor(max_workers=1) as pool:

                def send_model(**kwargs):
                    roots.append(Path(context[_TENSOR_DISK_OFFLOAD_ROOT_DIR]))
                    pending.append(pool.submit(adapter.call, future))
                    assert first_chunk.wait(5), "download did not reach the pause point"
                    kwargs["callback"](FLModel(params=tensors))

                controller.send_model = send_model
                try:
                    controller.run()
                    assert context == previous_context
                    assert not roots[0].exists()
                    # Root-scoped cleanup must leave another workflow's download usable.
                    assert other._cancel_reason is None
                    assert "T0" in consume(other, [chunks[0]], None)
                finally:
                    resume_download.set()
                    other.cleanup()
                pending[0].result(timeout=5)
            exit_process.assert_not_called()
            # Communicator cleanup happens only after the late chunk has been handled.
            comm.finalize_run(FLContext())
        else:
            adapter.call(future)
            exit_process.assert_not_called()

    if outcome == "success":
        callback.assert_called_once()
        payload = callback.call_args.args[0].payload
        for key, tensor in tensors.items():
            assert torch.equal(payload[key].materialize(), tensor)
        payload["T0"].release()
    else:
        callback.assert_not_called()
    assert not list(tmp_path.iterdir())
    if fails:
        cell.send_blob.assert_not_called()
    else:
        response = cell.send_blob.call_args.args[3]
        expected_rc = ReturnCode.OK if outcome == "success" else ReturnCode.SERVICE_UNAVAILABLE
        assert response.get_header(MessageHeaderKey.RETURN_CODE) == expected_rc
        assert response.get_header(MessageHeaderKey.REQ_ID) == "request-1"
        assert response.get_header(StreamHeaderKey.STREAM_REQ_ID) == "stream-1"
