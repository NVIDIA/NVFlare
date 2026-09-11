# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Client-only CoCo configuration; attestation is performed by Kata/Trustee."""

import json
import re
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.cc_constants import CCConfigKey, CCConfigValue
from nvflare.lighter.constants import PropKey, ProvFileName
from nvflare.lighter.spec import Builder

AUTHOR_PATH = "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer"
MANAGER_PATH = "nvflare.app_opt.confidential_computing.cc_manager.CCManager"


def resolve_cc_config(project, value):
    if not isinstance(value, str) or not value:
        raise ValueError("cc_config must be a non-empty YAML path")
    path = Path(value)
    if not path.is_absolute():
        project_file = project.get_prop("_project_file")
        path = (Path(project_file).parent if project_file else Path.cwd()) / path
    return str(path.resolve())


def validate_coco_config(config):
    allowed = {
        "compute_env",
        "cc_cpu_mechanism",
        CCConfigKey.CC_GPU,
        "role",
        "image_build",
        "release_name",
        "registry_repository",
        "platform_config",
        "class_allow_list",
        "cc_issuers",
        "cc_attestation",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unsupported CoCo configuration fields")
    for key, value in {
        "compute_env": CCConfigValue.CONFIDENTIAL_CONTAINERS,
        "cc_cpu_mechanism": CCConfigValue.AMD_SEV_SNP,
        CCConfigKey.CC_GPU: "nvidia",
        "role": "client",
    }.items():
        if config.get(key) != value:
            raise ValueError(f"CoCo requires {key}: {value}")
    image = config.get("image_build")
    if not isinstance(image, dict) or set(image) != {"context", "dockerfile"}:
        raise ValueError("image_build requires exactly context and dockerfile")
    for key, value in {**image, "platform_config": config.get("platform_config")}.items():
        if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
            raise ValueError(f"Invalid {key} path")
    release = config.get("release_name")
    if not isinstance(release, str) or len(release) > 63 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", release):
        raise ValueError("release_name must be a unique lowercase DNS label")
    repo = config.get("registry_repository")
    if not isinstance(repo, str) or not re.fullmatch(r"[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*", repo):
        raise ValueError("Invalid registry_repository")
    issuers = config.get("cc_issuers")
    if not isinstance(issuers, list) or len(issuers) != 1:
        raise ValueError("CoCo requires exactly one cc_issuers entry")
    issuer = issuers[0]
    if (
        not isinstance(issuer, dict)
        or set(issuer) != {"id", "path", "token_expiration", "args"}
        or issuer["id"] != "coco_authorizer"
        or issuer["path"] != AUTHOR_PATH
    ):
        raise ValueError("CoCo requires the coco_authorizer CoCoAuthorizer component")
    args = issuer["args"]
    if not isinstance(args, dict) or set(args) - {"trustee_public_key_file", "token_url"}:
        raise ValueError("Unsupported CoCo authorizer arguments")
    if not isinstance(args.get("trustee_public_key_file"), str) or not args["trustee_public_key_file"]:
        raise ValueError("trustee_public_key_file is required")
    age = issuer["token_expiration"]
    attestation = config.get("cc_attestation", {"check_frequency": 120})
    if not isinstance(attestation, dict) or set(attestation) != {"check_frequency"}:
        raise ValueError("cc_attestation requires exactly check_frequency")
    frequency = attestation["check_frequency"]
    if type(age) is not int or not 1 <= age <= 300 or type(frequency) is not int or not 0 < frequency < age:
        raise ValueError("Require 0 < check_frequency < token_expiration <= 300")


class CoCoBuilder(Builder):
    def initialize(self, project, ctx):
        packager = project.get_prop("packager", {})
        if packager.get("path") != "nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager":
            raise ValueError("CoCo clients require CoCoPackager; plaintext kits must not be released")
        releases = set()
        self.settings = {}
        for participant in project.get_all_participants():
            config = participant.get_prop(PropKey.CC_CONFIG_DICT, {})
            if config.get(CCConfigKey.COMPUTE_ENV) != CCConfigValue.CONFIDENTIAL_CONTAINERS:
                continue
            if participant.type != "client":
                raise ValueError("CoCo provisioning currently supports clients only")
            validate_coco_config(config)
            if config["release_name"] in releases:
                raise ValueError("Each CoCo client needs a distinct release_name")
            releases.add(config["release_name"])
            issuer = config["cc_issuers"][0]
            cc_path = Path(resolve_cc_config(project, participant.get_prop(PropKey.CC_CONFIG)))
            pem = (cc_path.parent / issuer["args"]["trustee_public_key_file"]).read_text()
            public = serialization.load_pem_public_key(pem.encode())
            if not isinstance(public, ec.EllipticCurvePublicKey) or not isinstance(public.curve, ec.SECP256R1):
                raise ValueError("Trustee EAR key must be the authenticated P-256 AS signing public key")
            args = {
                "trustee_public_key": pem,
                "audience": "nvflare-coco:" + project.name,
                "max_token_age_seconds": issuer["token_expiration"],
            }
            # Validate endpoint and arguments before any signed kit is released.
            from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

            url = issuer["args"].get("token_url", "http://127.0.0.1:8006/aa/token")
            CoCoAuthorizer(**args, token_url=url)
            self.settings[participant.name] = (args, url, config.get("cc_attestation", {}).get("check_frequency", 120))
        verifier_settings = [(s[0], s[2]) for s in self.settings.values()]
        if any(s != verifier_settings[0] for s in verifier_settings[1:]):
            raise ValueError("CoCo clients must share the pinned AS key and attestation timing")

    def build(self, project, ctx):
        for client in project.get_clients():
            config = client.get_prop(PropKey.CC_CONFIG_DICT, {})
            if config.get(CCConfigKey.COMPUTE_ENV) != CCConfigValue.CONFIDENTIAL_CONTAINERS:
                continue
            resources = Path(ctx.get_local_dir(client)) / ProvFileName.RESOURCES_JSON_DEFAULT
            if not resources.is_file():
                raise RuntimeError("CoCoBuilder requires StaticFileBuilder before CCBuilder")
            args, url, frequency = self.settings[client.name]
            self._write(
                ctx, client, "coco_authorizer", AUTHOR_PATH, {**args, "site_name": client.name, "token_url": url}
            )
            self._write(
                ctx,
                client,
                "cc_manager",
                MANAGER_PATH,
                {
                    "cc_issuers_conf": [
                        {"issuer_id": "coco_authorizer", "token_expiration": args["max_token_age_seconds"]}
                    ],
                    "cc_verifier_ids": [],
                    "cc_enabled_sites": list(self.settings),
                    "verify_frequency": frequency,
                    "verify_peer_tokens": False,
                },
            )
        args, _, frequency = next(iter(self.settings.values()))
        server = project.get_server()
        self._write(ctx, server, "coco_authorizer", AUTHOR_PATH, {**args, "expected_workloads": {}})
        self._write(
            ctx,
            server,
            "cc_manager",
            MANAGER_PATH,
            {
                "cc_issuers_conf": [],
                "cc_verifier_ids": ["coco_authorizer"],
                "cc_enabled_sites": list(self.settings),
                "verify_frequency": frequency,
                "required_namespaces": ["x-trustee-coco"],
            },
        )

    @staticmethod
    def _write(ctx, participant, component_id, path, args):
        target = Path(ctx.get_local_dir(participant)) / f"{component_id}__p_resources.json"
        target.write_text(
            json.dumps({"components": [{"id": component_id, "path": path, "args": args}]}, indent=2) + "\n"
        )
