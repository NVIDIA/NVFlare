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

"""Fresh-process TCP and shutdown probe for receive cleanup."""

import json
import threading

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.timing_utils import WAIT_TIMEOUT, wait_for


def reliable_cleanup_flow():
    url = f"tcp://localhost:{get_open_ports(1)[0]}"
    receiver = CoreCell("cleanup_rx", url, secure=False, credentials={})
    sender = CoreCell("cleanup_tx", url, secure=False, credentials={})
    receive_stream = StreamCell(receiver)
    send_stream = StreamCell(sender)
    frames = {}
    received = []
    errors = []
    ack_count = []

    def save_final(message):
        if message.get_header(StreamHeaderKey.DATA_TYPE) == StreamDataType.FINAL:
            frames[message.get_header(StreamHeaderKey.STREAM_ID)] = Message(dict(message.headers), message.payload)

    def receive(future):
        try:
            received.append(bytes(future.result()))
        except Exception as error:
            errors.append(str(error))

    def observe_ack(message):
        ack_count.append(message.get_header(StreamHeaderKey.STREAM_ID))

    receive_stream.register_blob_cb("cleanup", "data", receive)
    sender.add_outgoing_request_filter(STREAM_CHANNEL, STREAM_DATA_TOPIC, save_final)
    sender.add_incoming_filter(STREAM_CHANNEL, STREAM_ACK_TOPIC, observe_ack)
    try:
        receiver.start()
        sender.start()
        wait_for(lambda: sender.is_cell_connected("cleanup_rx") and receiver.is_cell_connected("cleanup_tx"))
        futures = []
        for index in range(128):
            payload = bytes([index]) * 4096
            future = send_stream.send_blob("cleanup", "data", "cleanup_rx", Message(payload=payload), reliable=True)
            assert future.result(timeout=WAIT_TIMEOUT) == len(payload)
            futures.append(future)
        wait_for(lambda: len(received) + len(errors) == 128)
        assert errors == []
        assert sorted(received) == [bytes([index]) * 4096 for index in range(128)]
        wait_for(lambda: RxTask.get_cleanup_stats()["pending"] == 128)
        with RxTask.map_lock:
            tasks = list(RxTask.rx_task_map.values())
        assert all(task.completed and task.completed_task_ttl == 65 for task in tasks)
        assert len([thread for thread in threading.enumerate() if thread.name == "f3-rx-cleanup"]) == 1

        # Retry an actual final frame over TCP after its original sender finished.
        sid = futures[0].get_stream_id()
        previous_acks = ack_count.count(sid)
        result = sender.fire_and_forget(STREAM_CHANNEL, STREAM_DATA_TOPIC, "cleanup_rx", frames[sid])
        assert not result.get("cleanup_rx")
        wait_for(lambda: ack_count.count(sid) > previous_acks)
        assert len(received) == 128  # retry acknowledged without redelivery
        assert RxTask.get_cleanup_stats()["pending"] == 128
        print(json.dumps({"streams": 128, "retry_ack": True, "before_shutdown": RxTask.get_cleanup_stats()}))
    finally:
        try:
            sender.stop()
        finally:
            receiver.stop()
            # Exercise the registered MPM cleanup sequence, including draining
            # stream pools before discarding receive tasks and joining the worker.
            done = threading.Event()
            MainProcessMonitor._do_cleanup(done)
            assert done.is_set()
    assert RxTask.rx_task_map == {}
    assert not any(thread.name == "f3-rx-cleanup" for thread in threading.enumerate())
    assert RxTask.get_cleanup_stats()["queue_depth"] == 0
    print("MPM cleanup released all receive tasks and scheduler threads", flush=True)
