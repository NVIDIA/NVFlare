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
import multiprocessing
import threading

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.deadline_scheduler import DeadlineScheduler
from nvflare.fuel.f3.streaming.shutdown import shutdown_f3_streaming
from nvflare.fuel.f3.streaming.stream_const import (
    STREAM_ACK_TOPIC,
    STREAM_CHANNEL,
    STREAM_DATA_TOPIC,
    StreamDataType,
    StreamHeaderKey,
)
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.timing_utils import WAIT_TIMEOUT, ThreadGroup, stop_processes, wait_for


def reliable_cleanup_flow(mode="mpm"):
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
            if mode == "explicit":
                shutdown_f3_streaming()
                assert RxTask.rx_task_map == {}
                assert RxTask.get_cleanup_stats()["pending"] == 0
                assert not RxTask.get_cleanup_stats()["worker_alive"]
                shutdown_f3_streaming()
                print("Explicit F3 shutdown released receive cleanup", flush=True)
        finally:
            try:
                sender.stop()
            finally:
                receiver.stop()
                # Exercise registered exit cleanup even after explicit teardown.
                done = threading.Event()
                MainProcessMonitor._do_cleanup(done)
                assert done.is_set()
    assert RxTask.rx_task_map == {}
    assert not any(thread.name == "f3-rx-cleanup" for thread in threading.enumerate())
    assert RxTask.get_cleanup_stats()["queue_depth"] == 0
    print("MPM cleanup released all receive tasks and scheduler threads", flush=True)


def _forked_scheduler_child(scheduler, connection):
    try:
        done = threading.Event()
        scheduler.schedule(0, done.set)
        assert done.wait(WAIT_TIMEOUT)
        scheduler.shutdown()
        connection.send(scheduler.snapshot())
    finally:
        connection.close()


def fork_cleanup_flow():
    scheduler = DeadlineScheduler()
    release = threading.Event()
    held = threading.Event()
    context = multiprocessing.get_context("fork")
    receiving, sending = context.Pipe(duplex=False)
    child = context.Process(target=_forked_scheduler_child, args=(scheduler, sending))

    def hold_condition():
        with scheduler._condition:
            held.set()
            release.wait()

    try:
        scheduler.schedule(3600, lambda: None)
        done = threading.Event()
        scheduler.schedule(0, done.set)
        assert done.wait(WAIT_TIMEOUT)
        wait_for(lambda: scheduler.snapshot()["expired"] == 1)
        with ThreadGroup(release) as workers:
            workers.start(hold_condition)
            assert held.wait(WAIT_TIMEOUT)
            try:
                child.start()
                sending.close()
                assert receiving.poll(WAIT_TIMEOUT), "Forked scheduler did not service its new deadline"
                stats = receiving.recv()
                assert stats["pending"] == stats["queue_depth"] == 0
                assert not stats["worker_alive"]
                child.join(WAIT_TIMEOUT)
                assert child.exitcode == 0
            finally:
                if child.pid is not None:
                    stop_processes(child)
        assert scheduler.snapshot()["pending"] == 1
        assert scheduler.snapshot()["worker_alive"]
        print("Forked child serviced deadlines without changing parent cleanup", flush=True)
    finally:
        receiving.close()
        sending.close()
        scheduler.shutdown()
