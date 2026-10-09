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

"""Prepare private CPU-only A/B provisioning inputs; never learn trust from a host."""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def prepare(output, run_id, base_image, as_key, as_key_sha256, mrtd, platform, server, cc_project):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,35}", run_id):
        raise ValueError("run_id must be a unique lowercase label of at most 36 characters")
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[a-f0-9]{64}", base_image):
        raise ValueError("A reviewed application base pinned by sha256 is required")
    if not re.fullmatch(r"[a-f0-9]{96}", mrtd):
        raise ValueError("An independently approved MRTD is required")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", server) or server in {"site-1", "site-2", "site-observer"}:
        raise ValueError("Invalid trusted server DNS identity")
    key = Path(as_key).read_bytes()
    if hashlib.sha256(key).hexdigest() != as_key_sha256:
        raise ValueError("AS public-key file differs from authenticated fingerprint")
    public = serialization.load_pem_public_key(key)
    if not isinstance(public, ec.EllipticCurvePublicKey) or not isinstance(public.curve, ec.SECP256R1):
        raise ValueError("AS signing key must be P-256")
    platform = Path(platform).resolve(strict=True)
    cc_project = Path(cc_project).resolve(strict=True)
    from nvflare.lighter.cc_provision.config import load_project_config

    common_template = load_project_config(cc_project)
    # The generated topology has a different declaring directory. Preserve the
    # loader's absolute paths and omit its private source-location marker.
    common_template.pop("_config_path", None)
    trustees = [
        name for name, service in common_template["attestation_services"].items() if service["type"] == "trustee"
    ]
    registries = list(common_template.get("container_registries", {}))
    if len(trustees) != 1 or len(registries) != 1 or "coco" not in common_template.get("build_tools", {}):
        raise ValueError("Acceptance requires exactly one Trustee service, one registry, and one CoCo build tool")
    trustee_name, registry_name = trustees[0], registries[0]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    output = Path(output)
    old_umask = os.umask(0o077)
    try:
        output.mkdir(parents=True, exist_ok=False)
        for topology in ("a", "b"):
            root = output / topology
            root.mkdir()
            (root / "trustee-as-public.pem").write_bytes(key)
            protected = ["site-1", "site-2"] + (["server"] if topology == "b" else [])
            common = copy.deepcopy(common_template)
            common_trustee = common["attestation_services"][trustee_name]
            common_trustee["attestation_signing_public_key_file"] = "./trustee-as-public.pem"
            common_trustee["workload_constraints"] = {
                participant: {"cpu_tee": "tdx", "tdx_mr_td": mrtd} for participant in protected
            }
            (root / "cc_project.yml").write_text(yaml.safe_dump(common, sort_keys=False))
            participants = [dict(name=server, type="server", org="acceptance", fed_learn_port=8002)]
            if topology == "b":
                participants[0]["cc_config"] = "cc_server.yml"
            participants += [
                dict(name=site, type="client", org="acceptance", cc_config=f"cc_{site}.yml")
                for site in protected
                if site != "server"
            ]
            participants += [
                dict(name="site-observer", type="client", org="acceptance"),
                dict(name="admin@example.com", type="admin", org="acceptance", role="project_admin"),
            ]
            builders = [
                dict(path=f"nvflare.lighter.impl.{module}.{name}")
                for module, name in (
                    ("workspace", "WorkspaceBuilder"),
                    ("static_file", "StaticFileBuilder"),
                    ("cert", "CertBuilder"),
                )
            ]
            builders += [
                dict(path="nvflare.lighter.cc_provision.impl.cc.CCBuilder"),
                dict(path="observer_builder.ObserverBuilder"),
                dict(path="nvflare.lighter.impl.signature.SignatureBuilder"),
            ]
            project = dict(
                api_version=3,
                name=f"tdx-{run_id}-{topology}",
                description="CPU-only TDX acceptance",
                cc_project_config="cc_project.yml",
                participants=participants,
                builders=builders,
                packager=dict(path="nvflare.lighter.cc_provision.impl.cc_packager.CCPackager"),
            )
            (root / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False))
            for site in protected:
                context = root / site
                context.mkdir()
                shutil.copyfile(HERE / "application/tdx_acceptance.py", context / "tdx_acceptance.py")
                (context / "Dockerfile").write_text(
                    f"FROM {base_image}\nCOPY --chown=65532:65532 tdx_acceptance.py /local/custom/tdx_acceptance.py\nENV PYTHONPATH=/local/custom\n"
                )
                config = dict(
                    schema_version=1,
                    cc_deployment_mode="coco",
                    cpu_tee="intel_tdx",
                    gpu_tee="none",
                    attestation={"service": trustee_name},
                    class_allow_list=["tdx_acceptance.AcceptanceController", "tdx_acceptance.AcceptanceExecutor"],
                    workload={"source": {"type": "docker_build", "context": f"./{site}", "dockerfile": "Dockerfile"}},
                    coco={
                        "release_name": f"{run_id}-{topology}-{site}",
                        "registry": registry_name,
                        "registry_repository": f"acceptance/{run_id}/{topology}/{site}",
                        "platform_config_file": str(platform),
                    },
                )
                (root / f"cc_{site}.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        (output / "inputs.json").write_text(
            json.dumps(
                dict(
                    run_id=run_id,
                    git_commit=commit,
                    base_image=base_image,
                    application_sha256=hashlib.sha256(
                        (HERE / "application/tdx_acceptance.py").read_bytes()
                    ).hexdigest(),
                    as_public_key_sha256=as_key_sha256,
                    approved_mrtd=mrtd,
                    platform_config_sha256=hashlib.sha256(platform.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        os.umask(old_umask)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "run-id",
        "base-image",
        "as-key",
        "as-key-sha256",
        "mrtd",
        "platform",
        "server",
        "cc-project",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
