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


def prepare(output, run_id, base_image, as_key, as_key_sha256, mrtd, platform, server):
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
                participants=participants,
                builders=builders,
                packager=dict(
                    path="nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager",
                    args=dict(build_image_cmd=str(HERE.parent / "admin/build_coco_image.sh")),
                ),
            )
            (root / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False))
            for site in protected:
                context = root / site
                context.mkdir()
                shutil.copyfile(HERE / "application/tdx_acceptance.py", context / "tdx_acceptance.py")
                (context / "Dockerfile").write_text(
                    f"FROM {base_image}\nCOPY --chown=65532:65532 tdx_acceptance.py /local/custom/tdx_acceptance.py\nENV PYTHONPATH=/local/custom\n"
                )
                args = dict(
                    trustee_public_key_file="./trustee-as-public.pem",
                    token_url="http://127.0.0.1:8006/aa/token",
                    retry_max_attempts=10,
                    retry_initial_delay=1.0,
                    retry_max_delay=15.0,
                    retry_backoff_multiplier=2.0,
                    retry_jitter_ratio=0.5,
                    workload_constraints={p: dict(cpu_tee="tdx", tdx_mr_td=mrtd) for p in protected},
                )
                config = dict(
                    compute_env="confidential_containers",
                    cc_cpu_mechanism="intel_tdx",
                    cc_gpu="none",
                    role="server" if site == "server" else "client",
                    cc_issuers=[
                        dict(
                            id="coco_authorizer",
                            path="nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer",
                            token_expiration=300,
                            args=args,
                        )
                    ],
                    cc_attestation=dict(
                        check_frequency=120,
                        registration_token_timeout=300,
                        refresh_token_timeout=30,
                        get_token_request_timeout=45,
                    ),
                    image_build=dict(context=f"./{site}", dockerfile="Dockerfile"),
                    release_name=f"{run_id}-{topology}-{site}",
                    registry_repository=f"acceptance/{run_id}/{topology}/{site}",
                    platform_config=str(platform),
                    class_allow_list=["tdx_acceptance.AcceptanceController", "tdx_acceptance.AcceptanceExecutor"],
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
    for name in ("output", "run-id", "base-image", "as-key", "as-key-sha256", "mrtd", "platform", "server"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
