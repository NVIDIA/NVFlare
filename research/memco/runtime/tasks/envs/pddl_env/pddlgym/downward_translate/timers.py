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

import contextlib
import os
import sys
import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.start_clock = self._clock()

    def _clock(self):
        times = os.times()
        return times[0] + times[1]

    def __str__(self):
        return "[%.3fs CPU, %.3fs wall-clock]" % (self._clock() - self.start_clock, time.time() - self.start_time)


@contextlib.contextmanager
def timing(text, block=False):
    timer = Timer()
    if block:
        print("%s..." % text)
    else:
        print("%s..." % text, end=" ")
    sys.stdout.flush()
    yield
    if block:
        print("%s: %s" % (text, timer))
    else:
        print(timer)
    sys.stdout.flush()
