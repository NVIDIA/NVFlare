#!/usr/bin/env python3
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

"""Build and publish the public measurement collector on a trusted owner machine."""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNP_GUEST_VERSION = "0.9.2"
SNP_GUEST_SHA256 = "9f008af82c37ad1d152d6b6f22fe4bf892d45f278d8b2500f69591f238905481"


def prepare(config, directory):
    helper = runpy.run_path(str(HERE / "lib/installed-platform.py"))
    keys = [
        "CPU_TEE",
        "COLLECTOR_REGISTRY_REPOSITORY",
        "COLLECTOR_REGISTRY_CA",
        "COLLECTOR_REGISTRY_AUTH_FILE",
    ]
    env = helper["read_config"](Path(config).resolve(strict=True), keys)
    tee = env.get("CPU_TEE")
    if tee not in ("amd_sev_snp", "intel_tdx"):
        raise ValueError("Select CPU_TEE=amd_sev_snp or CPU_TEE=intel_tdx for the public collector")
    directory = Path(directory).absolute()
    if directory.exists():
        raise ValueError("Use a new collector publication output directory")
    os.umask(0o077)
    directory.mkdir(parents=True, mode=0o700)
    build = directory / "build"
    build.mkdir(mode=0o700)
    dockerfile = "Dockerfile.tdx" if tee == "intel_tdx" else "Dockerfile"
    collector = "collect-tdx-evidence.py" if tee == "intel_tdx" else "collect-snp-evidence.sh"
    for source, name in ((dockerfile, "Dockerfile"), (collector, collector)):
        shutil.copyfile(HERE / "rehearsal-collector" / source, build / name)
    if tee == "amd_sev_snp":
        binary = build / "snpguest"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--tlsv1.2",
                "--output",
                str(binary),
                "https://github.com/virtee/snpguest/releases/download/v" + SNP_GUEST_VERSION + "/snpguest",
            ],
            check=True,
        )
        if helper["digest"](binary) != SNP_GUEST_SHA256:
            raise ValueError("Downloaded SNP collector binary differs from its reviewed SHA256")
        binary.chmod(0o755)
    # Only fixed public collector sources and the pinned public SNP binary enter the image build.
    (build / ".dockerignore").write_text("*\n!Dockerfile\n!" + collector + "\n!snpguest\n")
    local = "localhost/coco-" + tee + "-collector:" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    subprocess.run(["docker", "build", "--pull", "--tag", local, str(build)], check=True)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", local], text=True).strip()
    tag = env["COLLECTOR_REGISTRY_REPOSITORY"] + "-" + tee + ":" + local.rsplit(":", 1)[1]
    image = helper["publish_collector"](local, tag, directory, env)
    metadata = {
        "schema": 1,
        "cpu_tee": tee,
        "image": image,
        "image_id": image_id,
        "sources": {name: helper["digest"](build / name) for name in ("Dockerfile", collector)},
    }
    if tee == "amd_sev_snp":
        metadata["snpguest_sha256"] = SNP_GUEST_SHA256
    path = directory / "collector-image.json"
    with path.open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Collector image:", image)
    print("Metadata:", path)
    print("Optional metadata checksum:", helper["digest"](path))
    print("Transfer only the metadata and public registry CA to the selected worker; keep publisher credentials here.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    prepare(args.config, args.output_directory)
