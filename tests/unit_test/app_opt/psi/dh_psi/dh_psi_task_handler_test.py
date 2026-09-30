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
import logging
from typing import List

import pytest

# The DH-PSI modules require the optional openmined.psi dependency, which
# setup.cfg excludes on Python 3.14. Skip there rather than stub the import.
pytest.importorskip("private_set_intersection")

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_executor import PSIExecutor
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import DhPSITaskHandler, check_items_uniqueness

DUPLICATE_SENTINEL = "PRIVATE_PATIENT_SENTINEL"
OTHER_SENTINEL = "PRIVATE_RECORD_SENTINEL"


class _StaticItemsPSI(PSI):
    """Local PSI component that returns a fixed item list."""

    def __init__(self, items: List[str]):
        super().__init__(psi_writer_id="")
        self._items = items

    def initialize(self, fl_ctx: FLContext):
        pass

    def finalize(self, fl_ctx: FLContext):
        pass

    def load_items(self) -> List[str]:
        return self._items


def _prepare_shareable() -> Shareable:
    shareable = Shareable()
    shareable[PSIConst.TASK_KEY] = PSIConst.TASK_PREPARE
    shareable[PSIConst.BLOOM_FILTER_FPR] = 1e-9
    return shareable


def _executor_with_items(items: List[str]) -> PSIExecutor:
    task_handler = DhPSITaskHandler(local_psi_id="local_psi")
    task_handler.local_psi_handler = _StaticItemsPSI(items)
    executor = PSIExecutor(psi_algo_id="psi_algo")
    executor.task_handler = task_handler
    return executor


class TestCheckItemsUniqueness:
    def test_unique_items_are_accepted(self):
        check_items_uniqueness([DUPLICATE_SENTINEL, OTHER_SENTINEL, "third"])

    def test_duplicates_are_rejected(self):
        with pytest.raises(ValueError):
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL])

    def test_error_message_excludes_item_values(self):
        with pytest.raises(ValueError) as exc_info:
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL, OTHER_SENTINEL, OTHER_SENTINEL])

        message = str(exc_info.value)
        assert DUPLICATE_SENTINEL not in message
        assert OTHER_SENTINEL not in message
        assert message == "the items must be unique, found 2 items with duplicates"


class TestPSIExecutorDuplicateInput:
    """Drives duplicate input through the real executor and task handler."""

    def test_duplicate_input_is_rejected_without_logging_items(self, caplog, monkeypatch):
        # Secure logging is opt-in. Force it off so the full traceback reaches
        # the log; with it enabled the traceback is sanitized and this test
        # would pass even if the item values were reintroduced.
        monkeypatch.delenv("NVFLARE_SECURE_LOGGING", raising=False)

        executor = _executor_with_items([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL, OTHER_SENTINEL])

        with caplog.at_level(logging.DEBUG):
            reply = executor.execute(PSIConst.TASK, _prepare_shareable(), FLContext(), Signal())

        assert reply.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
        assert DUPLICATE_SENTINEL not in caplog.text
        assert OTHER_SENTINEL not in caplog.text

    def test_unique_input_is_accepted_by_the_executor(self, caplog, monkeypatch):
        monkeypatch.delenv("NVFLARE_SECURE_LOGGING", raising=False)

        executor = _executor_with_items([DUPLICATE_SENTINEL, OTHER_SENTINEL])

        with caplog.at_level(logging.DEBUG):
            reply = executor.execute(PSIConst.TASK, _prepare_shareable(), FLContext(), Signal())

        assert reply.get_return_code() == ReturnCode.OK
