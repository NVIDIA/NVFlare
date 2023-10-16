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
from typing import List, Optional

from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.data_exchange.data_exchanger import DataExchanger
from nvflare.app_common.decomposers import common_decomposers as app_common_decomposers
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor
from nvflare.fuel.utils.pipe.file_pipe import FilePipe


class FilePipeDataExchanger(DataExchanger):
    def __init__(
        self,
        data_exchange_path: str,
        supported_topics: List[str],
        file_accessor: Optional[FileAccessor] = None,
        pipe_name: str = "pipe",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Initializes the FilePipeModelExchanger.

        Args:
            data_exchange_path (str): The path for data exchange. This is the location where the data
                will be read from or written to.
            supported_topics (list[str]): Supported topics for data exchange. This allows the sender and receiver to identify
                the purpose or content of the data being exchanged.
            file_accessor (Optional[FileAccessor]): The file accessor for reading and writing files.
                If not provided, the default file accessor (FobsFileAccessor) will be used.
                Please refer to the docstring of the FileAccessor class for more information
                on implementing a custom file accessor. Defaults to None.
            pipe_name (str): The name of the pipe to be used for communication. This pipe will be used
                for transmitting data between the sender and receiver. Defaults to "pipe".
            get_poll_interval (float): The interval (in seconds) for checking if the other side has sent data.
                This determines how often the receiver checks for incoming data. Defaults to 0.5.
            read_interval (float): The interval (in seconds) for reading from the pipe. This determines
                how often the receiver reads data from the pipe. Defaults to 0.1.
            heartbeat_interval (float): The interval (in seconds) for sending heartbeat signals to the peer.
                Heartbeat signals are used to indicate that the sender or receiver is still active. Defaults to 5.0.
            heartbeat_timeout (float): The timeout (in seconds) for waiting for a heartbeat signal from the peer.
                If a heartbeat is not received within this timeout period, the connection may be considered lost.
                Defaults to 30.0.
        """
        flare_decomposers.register()
        app_common_decomposers.register()
        data_exchange_path = os.path.abspath(data_exchange_path)
        file_pipe = FilePipe(Mode.PASSIVE, data_exchange_path)
        if file_accessor is not None:
            file_pipe.set_file_accessor(file_accessor)

        super().__init__(
            supported_topics=supported_topics,
            pipe=file_pipe,
            pipe_name=pipe_name,
            get_poll_interval=get_poll_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
