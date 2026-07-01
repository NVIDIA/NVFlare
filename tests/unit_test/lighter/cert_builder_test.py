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

import datetime
from unittest.mock import patch

import pytest

from nvflare.lighter.constants import CtxKey, ParticipantType
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import load_crt


def _make_project():
    return Project(
        name="test-project",
        description="certificate validity test",
        participants=[
            Participant(type=ParticipantType.SERVER, name="server1", org="test_org"),
            Participant(type=ParticipantType.CLIENT, name="site-1", org="test_org"),
        ],
    )


def _cert_builder_from_project_config(root_valid_days):
    config = {
        "builders": [
            {
                "path": "nvflare.lighter.impl.cert.CertBuilder",
                "args": {"root_valid_days": root_valid_days},
            }
        ]
    }
    return prepare_builders(config)[0]


def _provision(workspace, cert_builder):
    provisioner = Provisioner(str(workspace), [WorkspaceBuilder(), cert_builder])
    return provisioner.provision(_make_project())


def _server_cert(workspace):
    return load_crt(str(workspace / "test-project" / "prod_00" / "server1" / "startup" / "server.crt"))


def test_default_root_validity_is_reported(tmp_path, capsys):
    workspace = tmp_path / "workspace"

    ctx = _provision(workspace, CertBuilder())

    assert not ctx.get(CtxKey.BUILD_ERROR)
    root_cert = ctx[CtxKey.ROOT_CERT]
    assert root_cert.not_valid_after_utc - root_cert.not_valid_before_utc == datetime.timedelta(days=360)
    assert (
        f"Root CA validity: NotBefore={root_cert.not_valid_before_utc.isoformat()}, "
        f"NotAfter={root_cert.not_valid_after_utc.isoformat()}" in capsys.readouterr().out
    )


def test_project_config_custom_root_validity_preserves_leaf_default(tmp_path):
    workspace = tmp_path / "workspace"

    ctx = _provision(workspace, _cert_builder_from_project_config(3650))

    assert not ctx.get(CtxKey.BUILD_ERROR)
    root_cert = ctx[CtxKey.ROOT_CERT]
    leaf_cert = _server_cert(workspace)
    assert root_cert.not_valid_after_utc - root_cert.not_valid_before_utc == datetime.timedelta(days=3650)
    assert leaf_cert.not_valid_after_utc - leaf_cert.not_valid_before_utc == datetime.timedelta(days=360)
    assert leaf_cert.not_valid_after_utc <= root_cert.not_valid_after_utc


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "3650", "ten years", None, [], {}])
def test_project_config_rejects_invalid_root_valid_days(value):
    with pytest.raises(ValueError, match="root_valid_days must be a positive integer"):
        _cert_builder_from_project_config(value)


def test_existing_root_validity_mismatch_fails_clearly(tmp_path):
    workspace = tmp_path / "workspace"
    first_ctx = _provision(workspace, CertBuilder())
    assert not first_ctx.get(CtxKey.BUILD_ERROR)

    second_ctx = _provision(workspace, CertBuilder(root_valid_days=3650))

    assert second_ctx.get(CtxKey.BUILD_ERROR)
    error = "\n".join(second_ctx.get_errors())
    assert "root_valid_days=3650 does not match the existing root CA certificate's actual validity of 360 days" in error
    assert "root_valid_days only controls a newly generated root" in error


def test_existing_root_with_matching_validity_is_reused(tmp_path):
    workspace = tmp_path / "workspace"
    first_ctx = _provision(workspace, _cert_builder_from_project_config(3650))

    second_ctx = _provision(workspace, _cert_builder_from_project_config(3650))

    assert not second_ctx.get(CtxKey.BUILD_ERROR)
    assert second_ctx[CtxKey.ROOT_CERT].serial_number == first_ctx[CtxKey.ROOT_CERT].serial_number


def test_leaf_validity_is_bounded_by_shorter_root(tmp_path):
    workspace = tmp_path / "workspace"
    builder = CertBuilder(root_valid_days=1)

    with patch.object(builder, "_generate_cert", wraps=builder._generate_cert) as generate_cert:
        ctx = _provision(workspace, builder)

    assert not ctx.get(CtxKey.BUILD_ERROR)
    root_cert = ctx[CtxKey.ROOT_CERT]
    leaf_cert = _server_cert(workspace)
    bounded_calls = [call for call in generate_cert.call_args_list if call.kwargs.get("not_valid_after")]
    assert bounded_calls
    assert all(call.kwargs["not_valid_before"] is not None for call in bounded_calls)
    assert leaf_cert.not_valid_before_utc == bounded_calls[0].kwargs["not_valid_before"].replace(microsecond=0)
    assert leaf_cert.not_valid_before_utc >= root_cert.not_valid_before_utc
    assert leaf_cert.not_valid_after_utc == root_cert.not_valid_after_utc
    assert leaf_cert.not_valid_after_utc - leaf_cert.not_valid_before_utc <= datetime.timedelta(days=1)
