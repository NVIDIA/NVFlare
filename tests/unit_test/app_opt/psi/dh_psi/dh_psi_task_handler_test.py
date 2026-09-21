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
import sys
from unittest.mock import MagicMock

import pytest

# dh_psi_task_handler imports the optional native private_set_intersection
# package, which is not installable in CI. The duplicate check under test does
# not use it, so stub the module before importing the handler.
sys.modules.setdefault("private_set_intersection", MagicMock())
sys.modules.setdefault("private_set_intersection.python", MagicMock())

from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import check_items_uniqueness  # noqa: E402
from nvflare.security.logging import secure_format_traceback  # noqa: E402

DUPLICATE_SENTINEL = "PRIVATE_PATIENT_SENTINEL"
OTHER_SENTINEL = "PRIVATE_RECORD_SENTINEL"


class TestCheckItemsUniqueness:
    def test_unique_items_are_accepted(self):
        check_items_uniqueness([DUPLICATE_SENTINEL, OTHER_SENTINEL, "third"])

    def test_duplicates_are_still_rejected(self):
        with pytest.raises(ValueError):
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL])

    def test_error_message_excludes_item_values(self):
        with pytest.raises(ValueError) as exc_info:
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL, OTHER_SENTINEL, OTHER_SENTINEL])

        message = str(exc_info.value)
        assert DUPLICATE_SENTINEL not in message
        assert OTHER_SENTINEL not in message

    def test_error_message_reports_duplicate_count(self):
        with pytest.raises(ValueError) as exc_info:
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL, OTHER_SENTINEL, OTHER_SENTINEL])

        message = str(exc_info.value)
        assert "2" in message
        # the count is reported on its own, not as a mapping of items to counts
        assert "{" not in message
        assert "}" not in message

    def test_traceback_excludes_item_values_when_secure_logging_is_off(self, monkeypatch):
        # Secure logging is opt-in, so an unset NVFLARE_SECURE_LOGGING is the
        # configuration in which log_exception records the full traceback.
        monkeypatch.delenv("NVFLARE_SECURE_LOGGING", raising=False)

        traceback_text = None
        try:
            check_items_uniqueness([DUPLICATE_SENTINEL, DUPLICATE_SENTINEL])
        except ValueError:
            traceback_text = secure_format_traceback()

        assert traceback_text is not None
        assert DUPLICATE_SENTINEL not in traceback_text
