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
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger
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


class TestModelExchanger:
    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get_fl_model_with_file_exchanger(self, weights):
        fl_model = FLModel(params=weights, params_type=ParamsType.FULL)
        test_pipe_name = "test_pipe"
        test_topic = "test_topic"
        flare_decomposers.register()
        common_decomposers.register()

        with tempfile.TemporaryDirectory() as root_dir:
            send_pipe = FilePipe(Mode.ACTIVE, root_path=root_dir)
            send_pipe.open(test_pipe_name)
            pipe_handler = PipeHandler(send_pipe)
            pipe_handler.start()
            req = Message.new_request(topic=test_topic, data=fl_model)
            _ = pipe_handler.send_to_peer(req)

            recv_pipe = FilePipe(Mode.PASSIVE, root_path=root_dir)
            y_mdx = ModelExchanger(pipe=recv_pipe, pipe_name=test_pipe_name, topic=test_topic)
            result_model = y_mdx.receive_global_model()

            for k, v in result_model.params.items():
                np.testing.assert_array_equal(weights[k], v)
            y_mdx.submit_model(result_model)

            start_time = time.time()
            receive_reply_model = None
            while True:
                if time.time() - start_time >= 50:
                    break
                reply: Optional[Message] = pipe_handler.get_next()
                if reply is not None and reply.topic == req.topic and req.msg_id == reply.req_id:
                    receive_reply_model = reply.data
                    break
                time.sleep(0.1)
            assert receive_reply_model is not None

            for k, v in receive_reply_model.params.items():
                np.testing.assert_array_equal(receive_reply_model.params[k], result_model.params[k])

            pipe_handler.stop(close_pipe=True)
            y_mdx.finalize()
