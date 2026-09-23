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

"""Server/client CoCo configuration; attestation is performed by Kata/Trustee."""

import json
import re
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.apis.fl_constant import SiteType
from nvflare.app_opt.confidential_computing.cc_timeouts import MANAGER_TIMEOUT_DEFAULTS, resolve_token_timeouts
from nvflare.lighter.cc_provision.cc_constants import CCConfigKey, CCConfigValue
from nvflare.lighter.cc_provision.utils import resolve_cc_config
from nvflare.lighter.constants import PropKey, ProvFileName
from nvflare.lighter.spec import Builder

AUTHOR_PATH = "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer"
MANAGER_PATH = "nvflare.app_opt.confidential_computing.cc_manager.CCManager"
COCO_STARTUP_PROLOGUE = "#!/usr/bin/env bash\nexec >/dev/null 2>&1\n"
RETRY_ARGUMENTS = {
    "retry_max_attempts",
    "retry_initial_delay",
    "retry_max_delay",
    "retry_backoff_multiplier",
    "retry_jitter_ratio",
}
VERIFIER_ARGUMENTS = {"proof_iat_leeway_seconds"}


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
    class_allow_list = config.get("class_allow_list", [])
    if not isinstance(class_allow_list, list) or any(
        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value)
        for value in class_allow_list
    ):
        raise ValueError(
            "CoCo class_allow_list requires explicit component class paths; wildcards/prefixes are forbidden"
        )
    for key, value in {
        "compute_env": CCConfigValue.CONFIDENTIAL_CONTAINERS,
        "cc_cpu_mechanism": CCConfigValue.AMD_SEV_SNP,
        CCConfigKey.CC_GPU: "nvidia",
    }.items():
        if config.get(key) != value:
            raise ValueError(f"CoCo requires {key}: {value}")
    if config.get("role") not in (SiteType.CLIENT, SiteType.SERVER):
        raise ValueError("CoCo role must be client or server")
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
    if not isinstance(args, dict) or set(args) - (
        {"trustee_public_key_file", "token_url"} | RETRY_ARGUMENTS | VERIFIER_ARGUMENTS
    ):
        raise ValueError("Unsupported CoCo authorizer arguments")
    if not isinstance(args.get("trustee_public_key_file"), str) or not args["trustee_public_key_file"]:
        raise ValueError("trustee_public_key_file is required")
    age = issuer["token_expiration"]
    attestation = config.get("cc_attestation", {"check_frequency": 120})
    if (
        not isinstance(attestation, dict)
        or "check_frequency" not in attestation
        or set(attestation) - ({"check_frequency"} | MANAGER_TIMEOUT_DEFAULTS.keys())
    ):
        raise ValueError("cc_attestation requires check_frequency and supports only token timeout options")
    frequency = attestation["check_frequency"]
    if type(age) is not int or not 1 <= age <= 300 or type(frequency) is not int or not 0 < frequency < age:
        raise ValueError("Require 0 < check_frequency < token_expiration <= 300")
    resolve_token_timeouts(**{name: attestation[name] for name in MANAGER_TIMEOUT_DEFAULTS if name in attestation})


class CoCoBuilder(Builder):
    emits_cc_manager = True
    is_exclusive = True

    def initialize(self, project, ctx):
        packager = project.get_prop("packager", {})
        if packager.get("path") != "nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager":
            raise ValueError("CoCo participants require CoCoPackager; plaintext kits must not be released")
        if any(client.name == SiteType.SERVER for client in project.get_clients()):
            raise ValueError("CoCo client name 'server' conflicts with the reserved server runtime identity")
        releases = set()
        self.settings = {}
        verifier_settings = []
        for participant in project.get_all_participants():
            config = participant.get_prop(PropKey.CC_CONFIG_DICT, {})
            if config.get(CCConfigKey.COMPUTE_ENV) != CCConfigValue.CONFIDENTIAL_CONTAINERS:
                continue
            validate_coco_config(config)
            if participant.type not in (SiteType.CLIENT, SiteType.SERVER) or config["role"] != participant.type:
                raise ValueError("CoCo role must match participant type (client or server)")
            if config["release_name"] in releases:
                raise ValueError("Each CoCo participant needs a distinct release_name")
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
                **{
                    name: value
                    for name, value in issuer["args"].items()
                    if name in RETRY_ARGUMENTS | VERIFIER_ARGUMENTS
                },
            }
            # Validate endpoint and arguments before any signed kit is released.
            from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

            url = issuer["args"].get("token_url", "http://127.0.0.1:8006/aa/token")
            authorizer = CoCoAuthorizer(**args, token_url=url)
            attestation = config.get("cc_attestation", {})
            timeouts = resolve_token_timeouts(
                **{name: attestation[name] for name in MANAGER_TIMEOUT_DEFAULTS if name in attestation}
            )
            self.settings[participant.name] = (args, url, attestation.get("check_frequency", 120), timeouts)
            # Compare effective verifier security settings, so an omitted
            # default and an explicitly configured default remain equivalent.
            verifier_settings.append(
                (
                    {
                        **{name: value for name, value in args.items() if name not in RETRY_ARGUMENTS},
                        "proof_iat_leeway_seconds": authorizer.proof_iat_leeway_seconds,
                    },
                    attestation.get("check_frequency", 120),
                    timeouts,
                )
            )
        if not self.settings:
            raise ValueError("CoCoBuilder requires at least one CoCo participant")
        if any(s != verifier_settings[0] for s in verifier_settings[1:]):
            raise ValueError("CoCo participants must share the pinned AS key and attestation timing")

    def build(self, project, ctx):
        server = project.get_server()
        protected_server = server.name in self.settings
        participants = [server, *project.get_clients()]
        # FLContext and CC envelopes use the root server's logical identity,
        # not its project hostname / certificate identity.
        enabled_sites = [
            SiteType.SERVER if participant.type == SiteType.SERVER else participant.name
            for participant in participants
            if participant.name in self.settings
        ]
        shared_args, _, shared_frequency, shared_timeouts = next(iter(self.settings.values()))
        for participant in participants:
            protected = participant.name in self.settings
            # Preserve client-only provisioning. When the server is protected,
            # ordinary clients must also verify it, without claiming to attest.
            if not protected and participant.type != SiteType.SERVER and not protected_server:
                continue
            resources = Path(ctx.get_local_dir(participant)) / ProvFileName.RESOURCES_JSON_DEFAULT
            if not resources.is_file():
                raise RuntimeError("CoCoBuilder requires StaticFileBuilder before CCBuilder")
            if protected:
                self._silence_startup(ctx, participant)
                args, url, frequency, timeouts = self.settings[participant.name]
                site = SiteType.SERVER if participant.type == SiteType.SERVER else participant.name
                authorizer_args = {**args, "site_name": site, "token_url": url}
                issuers = [{"issuer_id": "coco_authorizer", "token_expiration": args["max_token_age_seconds"]}]
            else:
                authorizer_args = {name: value for name, value in shared_args.items() if name not in RETRY_ARGUMENTS}
                frequency, timeouts, issuers = shared_frequency, shared_timeouts, []
            self._write(ctx, participant, "coco_authorizer", AUTHOR_PATH, authorizer_args)
            self._write(
                ctx,
                participant,
                "cc_manager",
                MANAGER_PATH,
                {
                    "cc_issuers_conf": issuers,
                    "cc_verifier_ids": ["coco_authorizer"],
                    "cc_enabled_sites": enabled_sites,
                    "required_site_verifier_ids": {site: ["coco_authorizer"] for site in enabled_sites},
                    "require_site_binding": True,
                    "verify_frequency": frequency,
                    **timeouts,
                },
            )

    @staticmethod
    def _silence_startup(ctx, participant):
        """Discard host-visible startup/process output before signing the kit."""
        kit = Path(ctx.get_ws_dir(participant))
        if (kit / ProvFileName.SIGNATURE_JSON).exists():
            raise RuntimeError("CoCoBuilder must run before SignatureBuilder")
        startup = kit / "startup/sub_start.sh"
        text = startup.read_text()
        shebang = "#!/usr/bin/env bash\n"
        if not text.startswith(shebang):
            raise RuntimeError("CoCoBuilder requires the standard Bash startup script")
        if not text.startswith(COCO_STARTUP_PROLOGUE):
            # Keep no duplicate of the original stdout/stderr descriptors. All
            # children inherit /dev/null; guest-local NVFlare file logs remain.
            startup.write_text(COCO_STARTUP_PROLOGUE + text[len(shebang) :])

    @staticmethod
    def _write(ctx, participant, component_id, path, args):
        target = Path(ctx.get_local_dir(participant)) / f"{component_id}__p_resources.json"
        target.write_text(
            json.dumps({"components": [{"id": component_id, "path": path, "args": args}]}, indent=2) + "\n"
        )
