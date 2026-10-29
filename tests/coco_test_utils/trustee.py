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

import hashlib
import json
from pathlib import Path

from nvflare.lighter.cc_provision import workload_release, workload_security

COCO = Path(__file__).resolve().parents[2] / "examples/devops/coco"

IMAGE = "registry.example.com:5443/workload@sha256:" + "a" * 64

INITIAL_POLICY = b'package policy\nimport rego.v1\ndefault allow := false\nallow if { input.fixture == "unrelated" }\n'


def update_manifest(directory):
    manifest = "".join(
        hashlib.sha256((directory / name).read_bytes()).hexdigest() + "  " + name + "\n"
        for name in sorted(workload_security.HANDOFF_FILES)
    )
    (directory / "SHA256SUMS").write_text(manifest)
    return hashlib.sha256(manifest.encode()).hexdigest()


def make_handoff(directory, runtime="kata-qemu-snp", release_name="demo"):
    directory.mkdir(parents=True)
    resources = [f"default/{kind}/{release_name}" for kind in ("image-key", "sig-public-key", "security-policy")]
    auth = workload_release.build_authorization(runtime, release_name, IMAGE, ["/bin/workload"], "b" * 64, resources)
    template = (COCO / "service/policies/workload-resource-policy.rego.template").read_text()
    (directory / "release-authorization.json").write_text(json.dumps(auth))
    (directory / "resource-policy-fragment.rego").write_text(workload_release.render_fragment(auth, template))
    (directory / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nfixture\n-----END PUBLIC KEY-----\n")
    (directory / "image_key").write_bytes(b"k" * 32)
    (directory / "image_key").chmod(0o600)
    (directory / "image-security-policy.json").write_text(
        json.dumps(
            {
                "default": [{"type": "reject"}],
                "transports": {
                    "docker": {
                        IMAGE.split("@")[0]: [
                            {
                                "type": "sigstoreSigned",
                                "keyPath": f"kbs:///default/sig-public-key/{release_name}",
                                "signedIdentity": {"type": "matchRepository"},
                            }
                        ]
                    }
                },
            }
        )
    )
    return directory, update_manifest(directory)
