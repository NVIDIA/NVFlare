# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.signal import Signal

_SMALL_WAIT = 0.01


class WaiterRC:
    OK = 0
    IS_SET = 1
    TIMEOUT = 2
    ABORTED = 3
    ERROR = 4


def conditional_wait(waiter: threading.Event, timeout: float, abort_signal: Signal, condition_cb=None, **cb_kwargs):
    wait_time = min(_SMALL_WAIT, timeout)
    start = time.time()
    while True:
        if waiter.wait(wait_time):
            # triggered
            return WaiterRC.IS_SET

        if time.time() - start >= timeout:
            return WaiterRC.TIMEOUT

        # check conditions
        if abort_signal and abort_signal.triggered:
            return WaiterRC.ABORTED

        if condition_cb:
            try:
                rc = condition_cb(**cb_kwargs)
                if rc != WaiterRC.OK:
                    return rc
            except:
                return WaiterRC.ERROR
