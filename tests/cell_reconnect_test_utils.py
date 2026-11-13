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

"""Spawned roles for the live Cell reconnect regression."""

import traceback

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic, MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.file_driver import FileDriver
from nvflare.fuel.f3.message import Message
from tests.timing_utils import wait_for

CHILD = "server.job"


def run_parent(config_dir, control, goodbye_seen, close_entered, close_released):
    cell = None
    armed = []
    original_close = FileDriver.close_connection

    def close_connection(driver, connection):
        if armed and connection is armed[0]:
            close_entered.set()
            if not close_released.wait(90):
                raise TimeoutError("test did not release the old connection's close notification")
        original_close(driver, connection)

    try:
        CommConfigurator.reload_from_dir(config_dir)
        cell = CoreCell("server", "tcp://127.0.0.1:0", False, {}, create_internal_listener=True)

        def goodbye(request):
            reply = cell._peer_goodbye(request)
            if request.get_header(MessageHeaderKey.ORIGIN) == CHILD:
                goodbye_seen.set()
            return reply

        cell.register_request_cb(CellChannel.CELLNET, CellChannelTopic.Bye, goodbye)
        FileDriver.close_connection = close_connection
        cell.start()
        control.send({"url": cell.get_internal_listener_url()})
        while True:
            command = control.recv()
            if command == "stop":
                break
            if command == "arm":
                connections = cell.communicator.conn_manager.get_connections(CHILD)
                assert len(connections) == 1
                armed.append(connections[0].conn)
                control.send({"armed": True})
            elif command == "request":
                reply = cell.send_request("reconnect", "echo", CHILD, Message(), timeout=3.0)
                control.send({"rc": reply.get_header(MessageHeaderKey.RETURN_CODE), "payload": reply.payload})
    except Exception:
        control.send({"error": traceback.format_exc()})
    finally:
        close_released.set()
        if cell:
            cell.stop()
        FileDriver.close_connection = original_close
        control.close()


def run_child(config_dir, parent_url, incarnation, ready, stop):
    cell = None
    try:
        CommConfigurator.reload_from_dir(config_dir)
        cell = CoreCell(CHILD, None, False, {}, parent_url=parent_url)
        cell.register_request_cb("reconnect", "echo", lambda request: Message(payload=incarnation))
        cell.start()
        wait_for(lambda: cell.is_cell_connected("server"))
        ready.put({"connected": incarnation})
        stop.wait(90)
    except Exception:
        ready.put({"error": traceback.format_exc()})
    finally:
        if cell:
            cell.stop()
