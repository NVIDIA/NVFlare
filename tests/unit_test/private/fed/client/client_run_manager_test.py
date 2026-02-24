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

from unittest.mock import MagicMock

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.private.fed.client.client_run_manager import ClientRunManager


class _DummyRunManager:
    def __init__(self):
        self.all_clients = None
        self.name_to_clients = {}


def test_get_job_clients_raises_if_job_clients_missing():
    run_manager = _DummyRunManager()
    fl_ctx = MagicMock()
    fl_ctx.get_prop.return_value = {}

    with pytest.raises(RuntimeError, match=f"missing {JobMetaKey.JOB_CLIENTS}"):
        ClientRunManager.get_job_clients(run_manager, fl_ctx)


def test_get_job_clients_raises_if_job_clients_not_list():
    run_manager = _DummyRunManager()
    fl_ctx = MagicMock()
    fl_ctx.get_prop.return_value = {JobMetaKey.JOB_CLIENTS: "bad"}

    with pytest.raises(RuntimeError, match=f"invalid {JobMetaKey.JOB_CLIENTS} type"):
        ClientRunManager.get_job_clients(run_manager, fl_ctx)


def test_get_job_clients_accepts_empty_list():
    run_manager = _DummyRunManager()
    fl_ctx = MagicMock()
    fl_ctx.get_prop.return_value = {JobMetaKey.JOB_CLIENTS: []}

    ClientRunManager.get_job_clients(run_manager, fl_ctx)

    assert run_manager.all_clients == []
    fl_ctx.set_prop.assert_called_once()
