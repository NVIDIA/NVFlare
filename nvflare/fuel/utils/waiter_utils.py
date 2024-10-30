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
    """Wait for an event until timeout, aborted, or some condition is met.

    Args:
        waiter: the event to wait
        timeout: the max time to wait
        abort_signal: signal to abort the wait
        condition_cb: condition to check during waiting
        **cb_kwargs: kwargs for the condition_cb

    Returns: return code to indicate how the waiting is stopped:
        IS_SET: the event is set
        TIMEOUT: the event timed out
        ABORTED: abort signal is triggered during the wait
        ERROR: the condition_cb encountered unhandled exception
        OK: only used by the condition_cb to say "all is normal"
        other integers: returned by condition_cb for other conditions met

    """
    wait_time = min(_SMALL_WAIT, timeout)
    start = time.time()
    while True:
        if waiter.wait(wait_time):
            # the event just happened!
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
                    # a bad condition is detected by the condition_cb
                    # we return the rc from the condition_cb
                    return rc
            except:
                return WaiterRC.ERROR
