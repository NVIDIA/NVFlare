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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.task_utils import apply_filters


def _filter():
    filter_ = MagicMock()
    filter_.process.side_effect = lambda data, fl_ctx: data
    return filter_


def test_apply_filters_without_configured_filters_returns_input():
    shareable = Shareable()

    result = apply_filters("unused", shareable, FLContext(), {}, "train", "in")

    assert result is shareable


def test_apply_filters_matches_all_tasks_wildcard():
    wildcard_filter = _filter()
    shareable = Shareable()

    result = apply_filters("unused", shareable, FLContext(), {"*/in": [wildcard_filter]}, "train", "in")

    assert result is shareable
    wildcard_filter.process.assert_called_once()


def test_apply_filters_matches_task_pattern():
    pattern_filter = _filter()
    shareable = Shareable()

    result = apply_filters("unused", shareable, FLContext(), {"train_*/in": [pattern_filter]}, "train_model", "in")

    assert result is shareable
    pattern_filter.process.assert_called_once()


def test_apply_filters_prefers_exact_match_over_patterns():
    wildcard_filter = _filter()
    pattern_filter = _filter()
    exact_filter = _filter()
    config_filters = {
        "*/in": [wildcard_filter],
        "train_*/in": [pattern_filter],
        "train_model/in": [exact_filter],
    }

    apply_filters("unused", Shareable(), FLContext(), config_filters, "train_model", "in")

    exact_filter.process.assert_called_once()
    pattern_filter.process.assert_not_called()
    wildcard_filter.process.assert_not_called()


def test_apply_filters_uses_first_matching_pattern():
    pattern_filter = _filter()
    wildcard_filter = _filter()
    config_filters = {
        "train_*/out": [pattern_filter],
        "*/out": [wildcard_filter],
    }

    apply_filters("unused", Shareable(), FLContext(), config_filters, "train_model", "out")

    pattern_filter.process.assert_called_once()
    wildcard_filter.process.assert_not_called()


def test_apply_filters_only_matches_requested_direction():
    inbound_filter = _filter()
    shareable = Shareable()

    result = apply_filters("unused", shareable, FLContext(), {"*/in": [inbound_filter]}, "train", "out")

    assert result is shareable
    inbound_filter.process.assert_not_called()
