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

import pytest

from nvflare.app_opt.confidential_computing.utils import NonceHistory


def test_nonce_history_rejects_invalid_and_replayed_values():
    history = NonceHistory(2)

    assert history.add(None) is False
    assert history.add("") is False
    assert history.add("nonce-1") is True
    assert history.add("nonce-1") is False


def test_nonce_history_evicts_oldest_value_at_configured_size():
    history = NonceHistory(2)

    assert history.add("nonce-1") is True
    assert history.add("nonce-2") is True
    assert history.add("nonce-3") is True
    assert history.add("nonce-1") is True


@pytest.mark.parametrize("size", [0, -1, 1.5, None])
def test_nonce_history_requires_positive_integer_size(size):
    with pytest.raises(ValueError):
        NonceHistory(size)
