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

"""Hold an optional file operation until lifecycle assertions finish."""

import threading
from contextlib import contextmanager
from unittest.mock import patch

from tests.timing_utils import ThreadGroup


@contextmanager
def stalled_resource_io(module, name, value=None):
    entered, release = threading.Event(), threading.Event()
    with ThreadGroup(release) as threads:

        def blocked(*args, **kwargs):
            threads.threads.append(threading.current_thread())
            entered.set()
            release.wait()
            return value

        with patch.object(module, name, blocked):
            yield entered, threads
