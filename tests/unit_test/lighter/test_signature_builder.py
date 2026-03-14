# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, call, patch

import pytest

from nvflare.lighter.constants import CtxKey, PropKey
from nvflare.lighter.impl.signature import SignatureBuilder


def _make_participant(cc_enabled):
    p = MagicMock()
    p.get_prop.side_effect = lambda key: cc_enabled if key == PropKey.CC_ENABLED else None
    return p


def _make_ctx(ws_dir, kit_dir, local_dir):
    ctx = MagicMock()
    ctx.get.side_effect = lambda key: "fake_key" if key == CtxKey.ROOT_PRI_KEY else None
    ctx.get_ws_dir.return_value = ws_dir
    ctx.get_kit_dir.return_value = kit_dir
    ctx.get_local_dir.return_value = local_dir
    return ctx


class TestSignatureBuilder:
    def test_missing_root_pri_key_raises(self):
        builder = SignatureBuilder()
        project = MagicMock()
        ctx = MagicMock()
        ctx.get.return_value = None

        with pytest.raises(RuntimeError, match="missing.*root_pri_key"):
            builder.build(project, ctx)

    @patch("nvflare.lighter.impl.signature.sign_folders")
    def test_non_cc_signs_only_startup_and_local(self, mock_sign):
        builder = SignatureBuilder()
        participant = _make_participant(cc_enabled=False)
        project = MagicMock()
        project.get_all_participants.return_value = [participant]
        ctx = _make_ctx(ws_dir="/ws", kit_dir="/ws/startup", local_dir="/ws/local")

        builder.build(project, ctx)

        # must sign startup and local, never the root
        assert mock_sign.call_count == 2
        signed_dirs = [c.args[0] for c in mock_sign.call_args_list]
        assert "/ws/startup" in signed_dirs
        assert "/ws/local" in signed_dirs
        assert "/ws" not in signed_dirs

    @patch("nvflare.lighter.impl.signature.sign_folders")
    def test_cc_signs_root(self, mock_sign):
        builder = SignatureBuilder()
        participant = _make_participant(cc_enabled=True)
        project = MagicMock()
        project.get_all_participants.return_value = [participant]
        ctx = _make_ctx(ws_dir="/ws", kit_dir="/ws/startup", local_dir="/ws/local")

        builder.build(project, ctx)

        assert mock_sign.call_count == 1
        assert mock_sign.call_args.args[0] == "/ws"

    @patch("nvflare.lighter.impl.signature.sign_folders")
    def test_mixed_participants(self, mock_sign):
        """CC and non-CC participants in the same project are each handled correctly."""
        builder = SignatureBuilder()
        cc_participant = _make_participant(cc_enabled=True)
        plain_participant = _make_participant(cc_enabled=False)
        project = MagicMock()
        project.get_all_participants.return_value = [cc_participant, plain_participant]

        ctx = MagicMock()
        ctx.get.side_effect = lambda key: "fake_key" if key == CtxKey.ROOT_PRI_KEY else None
        ctx.get_ws_dir.side_effect = lambda p: "/cc_ws" if p is cc_participant else "/plain_ws"
        ctx.get_kit_dir.side_effect = lambda p: "/plain_ws/startup"
        ctx.get_local_dir.side_effect = lambda p: "/plain_ws/local"

        builder.build(project, ctx)

        signed_dirs = [c.args[0] for c in mock_sign.call_args_list]
        assert "/cc_ws" in signed_dirs          # CC: root signed
        assert "/plain_ws/startup" in signed_dirs  # non-CC: startup signed
        assert "/plain_ws/local" in signed_dirs    # non-CC: local signed
        assert "/plain_ws" not in signed_dirs      # non-CC: root NOT signed
        assert mock_sign.call_count == 3

    @patch("nvflare.lighter.impl.signature.sign_folders")
    def test_transfer_dir_not_signed_for_non_cc(self, mock_sign):
        """Regression: transfer/ must never be signed in non-CC mode (poc prepare bug)."""
        builder = SignatureBuilder()
        participant = _make_participant(cc_enabled=False)
        project = MagicMock()
        project.get_all_participants.return_value = [participant]
        ctx = _make_ctx(ws_dir="/ws", kit_dir="/ws/startup", local_dir="/ws/local")

        builder.build(project, ctx)

        signed_dirs = [c.args[0] for c in mock_sign.call_args_list]
        assert "/ws/transfer" not in signed_dirs
