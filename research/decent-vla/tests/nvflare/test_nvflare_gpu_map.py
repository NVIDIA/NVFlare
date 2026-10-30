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

"""fl.nvflare.gpu is a client->GPU map, one entry per client."""
from __future__ import annotations

import pytest

from decent_vla.runtime.nvflare.job import check_gpu_map


def test_no_map_is_fine():
    assert check_gpu_map(None, 8) == []
    assert check_gpu_map("", 8) == []


def test_one_entry_per_client_passes():
    assert check_gpu_map("0,1", 2) == ["0", "1"]
    assert check_gpu_map("0,1,0,1,0,1,0,1", 8) == ["0", "1"] * 4


def test_single_group_is_left_alone():
    # one group = single-GPU juggling (or one client owning several cards);
    # NVFlare honours `threads` there, so a short map is not a mistake.
    assert check_gpu_map("0", 8) == ["0"]
    assert check_gpu_map("[0,1]", 8) == ["0,1"]


def test_more_groups_than_clients_is_left_to_nvflare():
    # NVFlare itself errors on this one; we must not shadow its message.
    assert check_gpu_map("0,1,0,1", 2) == ["0", "1", "0", "1"]


def test_short_map_is_rejected_with_the_fix_spelled_out():
    with pytest.raises(ValueError) as e:
        check_gpu_map("0,1", 8)
    msg = str(e.value)
    assert "2 GPU groups for 8 clients" in msg
    assert '"0,1,0,1,0,1,0,1"' in msg


def test_short_map_suggestion_cycles_uneven_groups():
    with pytest.raises(ValueError, match='"0,1,2,0,1"'):
        check_gpu_map("0,1,2", 5)
