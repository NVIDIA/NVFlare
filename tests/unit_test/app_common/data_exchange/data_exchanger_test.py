# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import tempfile
import time
from typing import Optional

import numpy as np
import pytest

from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.data_exchange.data_exchanger import DataExchanger
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Message
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(256, 256) for i in range(5)},
]

FLMODEL_TEST_CASES = [FLModel(params=x, params_type=ParamsType.FULL) for x in TEST_CASES]


@pytest.fixture
def setup_common(request):
    test_pipe_name = "test_pipe"
    test_topic = "test_topic"
    flare_decomposers.register()
    common_decomposers.register()
    with tempfile.TemporaryDirectory() as root_dir:
        send_pipe = FilePipe(Mode.ACTIVE, root_path=root_dir)
        send_pipe.open(test_pipe_name)
        pipe_handler = PipeHandler(send_pipe)
        pipe_handler.start()

        recv_pipe = FilePipe(Mode.PASSIVE, root_path=root_dir)
        y_dx = DataExchanger(pipe=recv_pipe, pipe_name=test_pipe_name, supported_topics=[test_topic])

        yield pipe_handler, test_topic, y_dx
        pipe_handler.stop(close_pipe=True)
        y_dx.finalize()


def get_reply(pipe_handler: PipeHandler, req: Message, timeout: float = 50.0) -> Optional[Message]:
    start_time = time.time()
    receive_reply = None
    while True:
        if time.time() - start_time >= timeout:
            break
        reply: Optional[Message] = pipe_handler.get_next()
        if reply is not None and reply.topic == req.topic and req.msg_id == reply.req_id:
            receive_reply = reply
            break
        time.sleep(0.1)
    return receive_reply


def check_fl_model_equivalence(a: FLModel, b: FLModel):
    assert a.params_type == b.params_type
    for k, v in a.params.items():
        np.testing.assert_array_equal(v, b.params[k])


def check_task_equivalence(a: ExchangeTask, b: ExchangeTask):
    assert a.task_id == b.task_id
    assert a.task_name == b.task_name
    if isinstance(a.data, FLModel):
        check_fl_model_equivalence(a.data, b.data)
    else:
        for k, v in a.data.items():
            np.testing.assert_array_equal(v, b.data[k])


class TestDataExchanger:
    @pytest.mark.parametrize("input_data", FLMODEL_TEST_CASES)
    def test_put_get_fl_model_with_file_exchanger(self, input_data, setup_common):
        pipe_handler, test_topic, y_dx = setup_common

        req = Message.new_request(topic=test_topic, data=input_data)
        _ = pipe_handler.send_to_peer(req)

        topic, received_data = y_dx.receive_data()
        assert topic == test_topic
        check_fl_model_equivalence(input_data, received_data)

        y_dx.submit_data(received_data)

        receive_reply = get_reply(pipe_handler, req)
        assert receive_reply is not None
        check_fl_model_equivalence(received_data, receive_reply.data)

    @pytest.mark.parametrize("input_data", TEST_CASES + FLMODEL_TEST_CASES)
    def test_put_get_exchange_task_with_file_exchanger(self, input_data, setup_common):
        pipe_handler, test_topic, y_dx = setup_common

        input_data = ExchangeTask(task_name="test_task", task_id="test_task_id", data=input_data)
        req = Message.new_request(topic=test_topic, data=input_data)
        _ = pipe_handler.send_to_peer(req)

        topic, received_data = y_dx.receive_data()
        assert topic == test_topic
        check_task_equivalence(input_data, received_data)

        y_dx.submit_data(received_data)

        received_reply = get_reply(pipe_handler, req)
        assert received_reply is not None
        assert received_reply.data is not None
        check_task_equivalence(received_data, received_reply.data)
