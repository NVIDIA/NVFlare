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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.collab.runtime.flare.executor import CollabExecutor


def test_get_site_index_uses_ordered_job_clients():
    clients = [Client("site-1", "token-1"), Client("site-2", "token-2")]

    assert CollabExecutor._get_site_index("site-2", clients) == 1

    with pytest.raises(RuntimeError, match="client missing is not present"):
        CollabExecutor._get_site_index("missing", clients)


def test_start_subprocess_worker_forwards_site_index():
    executor = CollabExecutor(client_obj_id="client", inprocess=False, training_module="training.module")
    executor.client_app = SimpleNamespace(obj=object())

    with patch("nvflare.collab.runtime.flare.executor.SubprocessLauncher") as launcher_cls:
        launcher_cls.return_value.start.return_value = True
        executor._start_subprocess_worker(MagicMock(), "site-2", site_index=1)

    assert launcher_cls.call_args.kwargs["site_index"] == 1
