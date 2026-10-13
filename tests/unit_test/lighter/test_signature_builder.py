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

import os
from unittest.mock import MagicMock, patch

import pytest

from nvflare.lighter.constants import PropKey, ProvFileName
from nvflare.lighter.impl.signature import SignatureBuilder


def _make_ctx(kit_dir, ws_dir=None, local_dir=None, root_pri_key="fake_key"):
    ctx = MagicMock()
    ctx.get.return_value = root_pri_key
    ctx.get_kit_dir.return_value = kit_dir
    ctx.get_ws_dir.return_value = ws_dir or kit_dir
    ctx.get_local_dir.return_value = local_dir or os.path.join(kit_dir, "local")
    return ctx


def _make_participant(cc_enabled=False, external_cert=False):
    p = MagicMock()
    p.get_prop.side_effect = {PropKey.CC_ENABLED: cc_enabled, PropKey.EXTERNAL_CERT: external_cert}.get
    return p


def _make_project(participants):
    proj = MagicMock()
    proj.get_all_participants.return_value = participants
    return proj


def test_plain_noncc_no_signature_json(tmp_path):
    """Plain non-CC, non-HE kit: sign_folders must NOT be called."""
    kit_dir = str(tmp_path / "kit")
    os.makedirs(kit_dir)
    ctx = _make_ctx(kit_dir)
    proj = _make_project([_make_participant(cc_enabled=False)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        mock_sign.assert_not_called()


def test_cc_enabled_generates_signature_json(tmp_path):
    """CC kit: sign_folders called for workspace dir."""
    ws_dir = str(tmp_path / "ws")
    kit_dir = str(tmp_path / "kit")
    os.makedirs(ws_dir)
    os.makedirs(kit_dir)
    ctx = _make_ctx(kit_dir, ws_dir=ws_dir)
    proj = _make_project([_make_participant(cc_enabled=True)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        mock_sign.assert_called_once_with(ws_dir, "fake_key", signature_file=ProvFileName.SIGNATURE_JSON)


def test_he_server_context_generates_signature_json(tmp_path):
    """HE kit with server_context.tenseal: sign_folders called."""
    kit_dir = str(tmp_path / "kit")
    local_dir = str(tmp_path / "local")
    os.makedirs(kit_dir)
    os.makedirs(local_dir)
    open(os.path.join(kit_dir, ProvFileName.SERVER_CONTEXT_TENSEAL), "w").close()
    ctx = _make_ctx(kit_dir, local_dir=local_dir)
    proj = _make_project([_make_participant(cc_enabled=False)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        assert mock_sign.call_count == 2
        mock_sign.assert_any_call(kit_dir, "fake_key", signature_file=ProvFileName.SIGNATURE_JSON)
        mock_sign.assert_any_call(local_dir, "fake_key", signature_file=ProvFileName.SIGNATURE_JSON)


def test_he_client_context_generates_signature_json(tmp_path):
    """HE kit with client_context.tenseal: sign_folders called."""
    kit_dir = str(tmp_path / "kit")
    local_dir = str(tmp_path / "local")
    os.makedirs(kit_dir)
    os.makedirs(local_dir)
    open(os.path.join(kit_dir, ProvFileName.CLIENT_CONTEXT_TENSEAL), "w").close()
    ctx = _make_ctx(kit_dir, local_dir=local_dir)
    proj = _make_project([_make_participant(cc_enabled=False)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        assert mock_sign.call_count == 2


def test_no_tenseal_files_no_signature_json(tmp_path):
    """Non-CC kit with no TenSEAL files: sign_folders NOT called."""
    kit_dir = str(tmp_path / "kit")
    os.makedirs(kit_dir)
    ctx = _make_ctx(kit_dir)
    proj = _make_project([_make_participant(cc_enabled=False)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        mock_sign.assert_not_called()


def test_missing_root_pri_key_raises(tmp_path):
    """Missing ROOT_PRI_KEY raises RuntimeError."""
    ctx = MagicMock()
    ctx.get.return_value = None
    proj = _make_project([_make_participant()])
    builder = SignatureBuilder()

    with pytest.raises(RuntimeError, match="missing"):
        builder.build(proj, ctx)


@pytest.mark.parametrize(
    "kit_type", ["plain", "cc", ProvFileName.SERVER_CONTEXT_TENSEAL, ProvFileName.CLIENT_CONTEXT_TENSEAL]
)
def test_external_cert_only_supported_for_unsigned_kits(tmp_path, kit_type):
    if kit_type.endswith(".tenseal"):
        (tmp_path / kit_type).touch()
    project = _make_project([_make_participant(cc_enabled=kit_type == "cc", external_cert=True)])
    with patch("nvflare.lighter.impl.signature.sign_folders") as sign:
        if kit_type == "plain":
            SignatureBuilder().build(project, _make_ctx(str(tmp_path)))
        else:
            with pytest.raises(ValueError, match="external_cert.*signed HE/CC"):
                SignatureBuilder().build(project, _make_ctx(str(tmp_path)))
        sign.assert_not_called()


def test_he_both_contexts_generates_signature_json(tmp_path):
    """HE kit with both context files: sign_folders still called (not doubled)."""
    kit_dir = str(tmp_path / "kit")
    local_dir = str(tmp_path / "local")
    os.makedirs(kit_dir)
    os.makedirs(local_dir)
    open(os.path.join(kit_dir, ProvFileName.SERVER_CONTEXT_TENSEAL), "w").close()
    open(os.path.join(kit_dir, ProvFileName.CLIENT_CONTEXT_TENSEAL), "w").close()
    ctx = _make_ctx(kit_dir, local_dir=local_dir)
    proj = _make_project([_make_participant(cc_enabled=False)])
    builder = SignatureBuilder()

    with patch("nvflare.lighter.impl.signature.sign_folders") as mock_sign:
        builder.build(proj, ctx)
        assert mock_sign.call_count == 2
