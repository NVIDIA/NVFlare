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

"""Exercise production builders for mixed CoCo targets without publishing images."""

import json
from pathlib import Path

import yaml

from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import verify_folder_signature
from tests.unit_test.lighter.cc_provision.impl.test_cc import _coco_project


def test_real_builders_sign_mixed_coco_kits(tmp_path):
    private = tmp_path / "mixed"
    private.mkdir()
    _coco_project(private)
    common = yaml.safe_load((private / "cc_project.yml").read_text())
    common["attestation_services"]["trustee"]["workload_constraints"] = {
        "site-1": {"cpu_tee": "tdx"},
        "site-2": {"cpu_tee": "snp"},
    }
    (private / "cc_project.yml").write_text(yaml.safe_dump(common, sort_keys=False))
    template = yaml.safe_load((private / "cc_site.yml").read_text())
    for name, cpu, gpu in (("site-1", "intel_tdx", "none"), ("site-2", "amd_sev_snp", "nvidia_cc")):
        config = {**template, "cpu_tee": cpu, "gpu_tee": gpu}
        config["coco"] = {
            **template["coco"],
            "release_name": name + "-v1",
            "registry_repository": "workloads/" + name,
        }
        (private / f"cc_{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    definition = {
        "api_version": 3,
        "name": "mixed_coco_project",
        "cc_project_config": "cc_project.yml",
        "participants": [
            {"name": "server.example.com", "type": "server", "org": "model_owner", "fed_learn_port": 8002},
            {"name": "site-1", "type": "client", "org": "data_owner_a", "cc_config": "cc_site-1.yml"},
            {"name": "site-2", "type": "client", "org": "data_owner_b", "cc_config": "cc_site-2.yml"},
            {"name": "admin@example.com", "type": "admin", "org": "model_owner", "role": "project_admin"},
        ],
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"config_folder": "config"}},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.cc_provision.impl.cc.CCBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
        "packager": {"path": "nvflare.lighter.cc_provision.impl.cc_packager.CCPackager"},
    }
    source = private / "project.yaml"
    source.write_text(yaml.safe_dump(definition, sort_keys=False))
    project = prepare_project(definition, project_file=str(source))
    # Disable packaging only in this offline test; real provisioning uses CCPackager.
    ctx = Provisioner(str(tmp_path / "workspace"), prepare_builders(definition)).provision(project)
    assert ctx.get(CtxKey.PROVISION_SUCCESS) is True, ctx.get_errors()
    result = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    assert {p.name for p in result.iterdir() if p.is_dir()} == {
        "server.example.com",
        "site-1",
        "site-2",
        "admin@example.com",
    }
    constraints = {
        "site-1": {"cpu_tee": "tdx", "gpu_required": False},
        "site-2": {"cpu_tee": "snp", "gpu_required": True},
    }
    for name in ("server.example.com", "site-1", "site-2"):
        kit = result / name
        manager = json.loads((kit / "local/cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = json.loads((kit / "local/trustee_authorizer__p_resources.json").read_text())["components"][0][
            "args"
        ]
        assert set(manager["cc_enabled_sites"]) == {"site-1", "site-2"}
        assert manager["required_site_verifier_ids"] == {site: ["trustee_authorizer"] for site in constraints}
        assert manager["require_site_binding"] is True
        assert manager["verify_frequency"] == 120
        assert manager["registration_token_timeout"] == 300
        assert manager["refresh_token_timeout"] == 30
        assert manager["get_token_request_timeout"] == 45
        assert authorizer["workload_constraints"] == constraints
        assert authorizer["audience"] == "nvflare-trustee:mixed_coco_project"
        if name == "server.example.com":
            assert manager["cc_issuers_conf"] == []
            assert "site_name" not in authorizer
        else:
            assert authorizer["site_name"] == name
            assert manager["cc_issuers_conf"] == [{"issuer_id": "trustee_authorizer", "token_expiration": 300}]
            assert (kit / "startup/sub_start.sh").read_text().startswith("#!/usr/bin/env bash\nexec >/dev/null 2>&1\n")
        if name != "server.example.com":
            assert verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
    protected = result / "site-1"
    resources = protected / "local/trustee_authorizer__p_resources.json"
    resources.write_text(resources.read_text() + " ")
    assert not verify_folder_signature(
        str(protected),
        str(protected / "startup/rootCA.pem"),
        single_signer=True,
        signature_file=ProvFileName.SIGNATURE_JSON,
    )
