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

import os
from typing import Optional

from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor
from nvflare.fuel.utils.pipe.file_pipe import FilePipe


class FilePipeModelExchanger(ModelExchanger):
    def __init__(
        self,
        data_exchange_path: str,
        file_accessor: Optional[FileAccessor] = None,
        pipe_name: str = "pipe",
        topic: str = "data",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Initializes the ModelExchanger.

        Args:
            data_exchange_path (str): Path for data exchange.
            file_accessor (Optional[FileAccessor]): File accessor for file operations. Defaults to None.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            topic (str): Topic for data exchange. Defaults to "data".
            get_poll_interval (float): Interval for checking if the other side has sent data. Defaults to 0.5.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
        """
        flare_decomposers.register()
        common_decomposers.register()
        data_exchange_path = os.path.abspath(data_exchange_path)
        file_pipe = FilePipe(Mode.PASSIVE, data_exchange_path)
        if file_accessor is not None:
            file_pipe.set_file_accessor(file_accessor)

        super().__init__(
            pipe=file_pipe,
            pipe_name=pipe_name,
            topic=topic,
            get_poll_interval=get_poll_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
