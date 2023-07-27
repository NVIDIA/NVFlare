# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import time


class Signal(object):
    def __init__(self, parent=None):
        """Init the Signal.

        Used to signal between and within FL Components.
        """
        self._value = None
        self._trigger_time = None
        self._triggered = False
        self._parent = parent

    def trigger(self, value):
        """Trigger the Signal.

        Args:
            value: set the value of the signal
        """
        self._value = value
        self._trigger_time = time.time()
        self._triggered = True

    @property
    def value(self):
        return self._value

    @property
    def trigger_time(self):
        return self._trigger_time

    def reset(self, value=None):
        """Reset the Signal.

        Args:
            value: reset the value of the signal
        """
        self._value = value
        self._trigger_time = None
        self._triggered = False

    @property
    def triggered(self):
        if self._triggered:
            return True
        if self._parent:
            return self._parent.triggered
        else:
            return False
