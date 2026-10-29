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


"""Shared factories for CoCo unit and integration contracts."""

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.impl.cc import CC_PACKAGER_PATH
from nvflare.lighter.provision import prepare_project


def _write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def _coco_project(tmp_path):
    for name in ("ca.pem", "admin.jwt", "registry-ca.pem", "username", "password"):
        (tmp_path / name).write_text("fixture")
    public = ec.generate_private_key(ec.SECP256R1()).public_key()
    (tmp_path / "as.pem").write_bytes(
        public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    command = tmp_path / "build"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o700)
    context = tmp_path / "site"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM reviewed-base\n")
    admin = tmp_path / "admin"
    admin.mkdir()
    (admin / "platform.env").write_text("# fixture\n")
    _write(
        tmp_path / "cc_project.yml",
        {
            "schema_version": 1,
            "attestation_services": {
                "trustee": {
                    "type": "trustee",
                    "kbs_endpoint": "https://trustee.example.com:8443",
                    "ca_cert_file": "ca.pem",
                    "admin_token_file": "admin.jwt",
                    "attestation_token_endpoint": "http://127.0.0.1:8006/aa/token",
                    "attestation_signing_public_key_file": "as.pem",
                    "token_expiration_seconds": 300,
                    "check_frequency_seconds": 120,
                    "registration_token_timeout_seconds": 300,
                    "refresh_token_timeout_seconds": 30,
                    "get_token_request_timeout_seconds": 45,
                    "proof_iat_leeway_seconds": 90,
                    "workload_constraints": {
                        "site-1": {
                            "cpu_tee": "snp",
                            "init_data": "a" * 64,
                        }
                    },
                }
            },
            "container_registries": {
                "workloads": {
                    "endpoint": "registry.example.com:5000",
                    "ca_cert_file": "registry-ca.pem",
                    "publisher_username_file": "username",
                    "publisher_password_file": "password",
                }
            },
            "build_tools": {"coco": {"build_command": "build"}},
        },
    )
    _write(
        tmp_path / "cc_site.yml",
        {
            "schema_version": 1,
            "cc_deployment_mode": "coco",
            "cpu_tee": "amd_sev_snp",
            "gpu_tee": "nvidia_cc",
            "attestation": {"service": "trustee"},
            "workload": {"source": {"type": "docker_build", "context": "site", "dockerfile": "Dockerfile"}},
            "coco": {
                "release_name": "site-v1",
                "registry": "workloads",
                "registry_repository": "workloads/site",
                "platform_config_file": "admin/platform.env",
            },
        },
    )
    project_dict = {
        "api_version": 3,
        "name": "coco-unified",
        "cc_project_config": "cc_project.yml",
        "participants": [
            {"type": "server", "name": "server.example.com", "org": "example"},
            {"type": "client", "name": "site-1", "org": "example", "cc_config": "cc_site.yml"},
        ],
        "packager": {"path": CC_PACKAGER_PATH},
    }
    return prepare_project(project_dict, project_file=tmp_path / "project.yml")
