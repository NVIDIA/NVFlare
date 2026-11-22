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

"""A signed Azure CC workspace must verify, including files written during finalization."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.provision import provision
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import verify_folder_signature

ROOT_CA = "rootCA.pem"


def _verified(workspace):
    return verify_folder_signature(
        str(workspace),
        str(workspace / "startup" / ROOT_CA),
        single_signer=True,
        signature_file=ProvFileName.SIGNATURE_JSON,
    )


class CCEnablingBuilder(Builder):
    """Stand in for the Azure CC builders by marking participants CC-enabled."""

    def build(self, project, ctx):
        for participant in project.get_all_participants():
            participant.set_prop(PropKey.CC_ENABLED, True)


def _config():
    return {
        "api_version": 3,
        "name": "ccproject",
        "description": "Azure CC signing regression",
        "participants": [
            {"name": "server.example.com", "type": "server", "org": "org", "fed_learn_port": 9002, "admin_port": 9003},
            {
                "name": "site-1",
                "type": "client",
                "org": "org",
                "connect_to": {"host": "server.example.com", "port": 9102},
                # Gives the client a comm_config.json, which StaticFileBuilder
                # only writes during finalize(); that file is the regression.
                "listening_host": {"port": 9200},
            },
            {"name": "admin@example.com", "type": "admin", "org": "org", "role": "project_admin"},
        ],
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "tests.unit_test.lighter.test_cc_signature.CCEnablingBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
    }


@pytest.fixture
def prod_dir(tmp_path):
    ctx = provision(
        SimpleNamespace(gen_scripts=False),
        copy.deepcopy(_config()),
        str(tmp_path / "project.yml"),
        str(tmp_path / "workspace"),
    )
    assert ctx[CtxKey.PROVISION_SUCCESS] is True
    return Path(ctx[CtxKey.CURRENT_PROD_DIR])


def test_finalization_output_is_covered_by_the_workspace_signature(prod_dir):
    """StaticFileBuilder writes local/comm_config.json in finalize(); it must be signed."""
    workspace = prod_dir / "site-1"
    comm_config = workspace / "local" / ProvFileName.COMM_CONFIG
    assert comm_config.is_file(), "the regression needs a file created during finalization"
    signatures = json.loads((workspace / "local" / ProvFileName.SIGNATURE_JSON).read_text())
    assert ProvFileName.COMM_CONFIG in signatures


@pytest.mark.parametrize("name", ["server.example.com", "site-1"])
def test_provisioned_cc_workspace_passes_startup_integrity_verification(prod_dir, name):
    assert _verified(prod_dir / name)


def test_tampering_with_a_signed_file_still_fails_verification(prod_dir):
    workspace = prod_dir / "site-1"
    comm_config = workspace / "local" / ProvFileName.COMM_CONFIG
    comm_config.write_text(comm_config.read_text() + "\n# tampered\n")
    assert not _verified(workspace)
