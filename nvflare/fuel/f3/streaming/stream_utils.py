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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nvflare.fuel.f3.connection import BytesAlike

STREAM_THREAD_POOL_SIZE = 128

stream_thread_pool = ThreadPoolExecutor(STREAM_THREAD_POOL_SIZE, "stm")
lock = threading.Lock()
start_time = time.time() * 1000000  # microseconds
stream_count = 0


def wrap_view(buffer: BytesAlike) -> memoryview:
    if isinstance(buffer, memoryview):
        view = buffer
    else:
        view = memoryview(buffer)

    return view


def gen_stream_id():
    global lock, stream_count, start_time
    with lock:
        stream_count += 1
    return f"SID{(start_time + stream_count):16.0f}"
