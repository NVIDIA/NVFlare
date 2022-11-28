# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from io import BytesIO
from typing import Optional, Any

from nvflare.fci.endpoint import Endpoint
from nvflare.fci.headers import Headers
from nvflare.fci.receivers import BytesReceiver, ObjectReceiver, StreamReceiver, FobsReceiver
from nvflare.fuel.utils import fobs


def get_endpoint(name: str) -> Endpoint:
    pass


def send_bytes(endpoint: Endpoint, channel: int, headers: Headers, payload: bytes):
    pass


def register_bytes_receiver(endpoint: Optional[Endpoint], channel: int, receiver: BytesReceiver):
    pass


def send_object(endpoint: Endpoint, channel: int, headers: Headers, data: Any):
    send_bytes(endpoint, channel, headers, fobs.dumps(data))


def register_object_receiver(endpoint: Optional[Endpoint], channel: int, receiver: ObjectReceiver):
    bytes_receiver = FobsReceiver(receiver)
    register_bytes_receiver(endpoint, channel, bytes_receiver)


def send_stream(endpoint: Endpoint, channel: int, headers: Headers) -> BytesIO:
    pass


def register_stream_receiver(endpoint: Optional[Endpoint], channel: int, receiver: StreamReceiver):
    pass
