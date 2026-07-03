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

"""Direct tests for the CellPipe alias grammar shared by naming, mTLS
identity resolution, and stream message authentication."""

import pytest

from nvflare.fuel.f3.cellnet.fqcn import make_cell_pipe_alias, parse_cell_pipe_alias


class TestCellPipeAliasGrammar:
    @pytest.mark.parametrize(
        "owner,runtime_id,mode",
        [
            ("site-1", "job-123", "active"),
            ("site-1", "8cb50f16-8158-46f6-a8d7-ec85b1f06c53", "passive"),
            ("site_a", "job", "active"),  # owners may contain "_"
        ],
    )
    def test_round_trip(self, owner, runtime_id, mode):
        alias = make_cell_pipe_alias(owner, runtime_id, mode)
        assert parse_cell_pipe_alias(alias) == (owner, runtime_id, mode)

    def test_owner_with_underscores_parses_from_the_right(self):
        # right-anchored parsing: the runtime id can never contain "_", so the
        # only valid owner of this alias is "site-a_x", never "site-a"
        assert parse_cell_pipe_alias("site-a_x_job-123_active") == ("site-a_x", "job-123", "active")

    @pytest.mark.parametrize(
        "segment",
        [
            "site-1_active",  # too few parts
            "site-1__active",  # empty runtime id
            "_job_active",  # empty owner
            "site-1_job-123_idle",  # unknown mode
            "site-1_job.x_active",  # "." in runtime id
            "job-123",  # no mode suffix
            "",
        ],
    )
    def test_invalid_segments_are_rejected(self, segment):
        assert parse_cell_pipe_alias(segment) is None
