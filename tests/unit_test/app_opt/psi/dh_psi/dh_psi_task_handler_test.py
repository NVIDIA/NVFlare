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
import importlib
import sys
from unittest.mock import MagicMock

import pytest

from nvflare.security.logging import secure_format_traceback

DUPLICATE_SENTINEL = "PRIVATE_PATIENT_SENTINEL"
OTHER_SENTINEL = "PRIVATE_RECORD_SENTINEL"

_STUBBED_MODULES = ("private_set_intersection", "private_set_intersection.python")


def _load_check_items_uniqueness():
    """Imports the duplicate check with the optional native dependency stubbed.

    dh_psi_task_handler imports PSIClient and PSIServer, which import the
    optional private_set_intersection package. That package is not installable
    in CI, so it is stubbed for the duration of this import only. Every module
    added to sys.modules by the import is then removed and the stubbed entries
    are restored, so later tests still observe the real environment.
    """
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    for name in _STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    before = set(sys.modules)
    try:
        module = importlib.import_module("nvflare.app_opt.psi.dh_psi.dh_psi_task_handler")
        return module.check_items_uniqueness
    finally:
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


check_items_uniqueness = _load_check_items_uniqueness()


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

        assert str(exc_info.value) == "the items must be unique, found 2 items with duplicates"

    def test_stubbed_dependency_is_not_left_in_sys_modules(self):
        for name in _STUBBED_MODULES:
            assert not isinstance(sys.modules.get(name), MagicMock)

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
