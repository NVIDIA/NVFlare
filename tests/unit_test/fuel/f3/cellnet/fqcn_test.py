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

from nvflare.fuel.f3.cellnet.fqcn import CELL_PIPE_ALIAS_PREFIX, make_cell_pipe_alias, parse_cell_pipe_alias


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
        assert alias.startswith(CELL_PIPE_ALIAS_PREFIX)
        assert parse_cell_pipe_alias(alias) == (owner, runtime_id, mode)

    def test_legacy_bare_alias_is_still_parsed(self):
        # pre-2.8 flat CellPipe names are whole-FQCN aliases with no prefix
        assert parse_cell_pipe_alias("site-1_job-123_active") == ("site-1", "job-123", "active")

    @pytest.mark.parametrize("segment", ["site-a_x_job-123_active", "cellpipe-alias-site-a_x_job-123_active"])
    def test_owner_with_underscores_parses_from_the_right(self, segment):
        # right-anchored parsing: the runtime id can never contain "_", so the
        # only valid owner of this alias is "site-a_x", never "site-a"
        assert parse_cell_pipe_alias(segment) == ("site-a_x", "job-123", "active")

    @pytest.mark.parametrize(
        "segment",
        [
            "site-1_active",  # too few parts
            "site-1__active",  # empty runtime id
            "_job_active",  # empty owner
            "site-1_job-123_idle",  # unknown mode
            "site-1_job.x_active",  # "." in runtime id
            "job-123",  # no mode suffix
            "cellpipe-alias-x_active",  # marked, but no owner/runtime split
            "cellpipe-alias-site-1_job-123_idle",  # marked, unknown mode
            "cellpipe-job-123_active",  # plain pipe leaf, no owner/runtime split
            "",
        ],
    )
    def test_invalid_segments_are_rejected(self, segment):
        assert parse_cell_pipe_alias(segment) is None

    def test_plain_leaf_with_underscore_token_bare_parses_so_callers_must_gate(self):
        # The bare legacy grammar cannot tell a plain leaf with an underscore
        # token from an alias - that is exactly why callers only accept the
        # bare form for whole-FQCN (single-segment) names and require the
        # cellpipe-alias- marker everywhere else.
        assert parse_cell_pipe_alias("cellpipe-simulate_job_active") == ("cellpipe-simulate", "job", "active")
