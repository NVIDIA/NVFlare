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

"""Target-aware workload authorization, rendered only with administrator-owned Rego.

The pinned Trustee TDX verifier regularizes SHA-256 InitData to 48 bytes for
MRCONFIGID and emits that complete value as lowercase-hex ``init_data``.
No caller-supplied encoding or trust-vector negotiation is permitted.
"""

import copy
import json
import re
import runpy
from pathlib import Path
from string import Template

if __package__:
    from nvflare.app_opt.confidential_computing.trustee_claims import CPU_TRUST_VECTORS, TRUST_VECTOR
else:
    _claims = runpy.run_path(str(Path(__file__).resolve().parent / "trustee_claims.py"))
    TRUST_VECTOR, CPU_TRUST_VECTORS = _claims["TRUST_VECTOR"], _claims["CPU_TRUST_VECTORS"]

RUNTIME_TARGETS = {
    "kata-qemu-nvidia-gpu-snp": ("snp", "nvidia"),
    "kata-qemu-snp": ("snp", "none"),
    "kata-qemu-nvidia-gpu-tdx": ("tdx", "nvidia"),
    "kata-qemu-tdx": ("tdx", "none"),
}
V1 = "coco-workload-owner-authorization/v1"
V2 = "coco-workload-owner-authorization/v2"
COMMON_FIELDS = {
    "schema",
    "release_name",
    "encrypted_image",
    "process_args",
    "required_ear_submods",
    "required_ear_trust_vectors",
    "kbs_resource_paths",
}
V1_FIELDS = COMMON_FIELDS | {"snp_init_data_sha256", "snp_init_data_encoding_in_trustee_v0_21"}
V2_FIELDS = COMMON_FIELDS | {"cpu_tee", "gpu", "init_data_sha256", "init_data_claim"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_authorization(path):
    return validate_authorization(json.loads(Path(path).read_text(), object_pairs_hook=unique_object))


def target_for_runtime(runtime):
    require(isinstance(runtime, str) and runtime in RUNTIME_TARGETS, "unsupported workload runtime")
    return RUNTIME_TARGETS[runtime]


def init_data_claim(cpu_tee, digest):
    require(cpu_tee in ("snp", "tdx"), "unsupported CPU TEE")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid SHA-256 InitData digest")
    # deps/verifier/src/tdx/{mod,claims}.rs at Trustee 338610fbfed57b66c61a8a3a60e0e4386bdce793.
    return digest + ("0" * 32 if cpu_tee == "tdx" else "")


def required_vectors(cpu_tee, gpu):
    require(cpu_tee in ("snp", "tdx") and gpu in ("none", "nvidia"), "unsupported CPU/GPU target")
    vectors = {"cpu0": dict(CPU_TRUST_VECTORS[cpu_tee])}
    if gpu == "nvidia":
        vectors["gpu0"] = dict(TRUST_VECTOR)
    return vectors


def build_authorization(runtime, release, image, args, digest, paths):
    cpu_tee, gpu = target_for_runtime(runtime)
    vectors = required_vectors(cpu_tee, gpu)
    return validate_authorization(
        {
            "schema": V2,
            "release_name": release,
            "encrypted_image": image,
            "process_args": args,
            "cpu_tee": cpu_tee,
            "gpu": gpu,
            "init_data_sha256": digest,
            "init_data_claim": init_data_claim(cpu_tee, digest),
            "required_ear_submods": list(vectors),
            "required_ear_trust_vectors": vectors,
            "kbs_resource_paths": paths,
        }
    )


def validate_authorization(authorization):
    """Validate exact owner data; return v2, normalizing only the legacy SNP+GPU contract."""
    require(isinstance(authorization, dict), "authorization must be an object")
    auth = copy.deepcopy(authorization)
    schema = auth.get("schema")
    require(schema in (V1, V2), "unsupported release-authorization schema")
    require(set(auth) == (V1_FIELDS if schema == V1 else V2_FIELDS), "unexpected release-authorization fields")
    if schema == V1:
        require(auth.pop("snp_init_data_encoding_in_trustee_v0_21") == "lowercase-hex", "unexpected SNP encoding")
        digest = auth.pop("snp_init_data_sha256")
        auth.update(schema=V2, cpu_tee="snp", gpu="nvidia", init_data_sha256=digest, init_data_claim=digest)
    release = auth["release_name"]
    require(
        isinstance(release, str) and re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", release) and len(release) <= 63,
        "invalid release_name",
    )
    image = auth["encrypted_image"]
    require(isinstance(image, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image), "immutable image required")
    args = auth["process_args"]
    require(
        isinstance(args, list)
        and args
        and all(isinstance(v, str) and v and "\0" not in v for v in args)
        and args[0].startswith("/"),
        "absolute process executable and nonempty string arguments required",
    )
    expected = init_data_claim(auth["cpu_tee"], auth["init_data_sha256"])
    require(auth["init_data_claim"] == expected, "InitData claim does not match the pinned verifier encoding")
    vectors = required_vectors(auth["cpu_tee"], auth["gpu"])
    require(auth["required_ear_submods"] == list(vectors), "incorrect required EAR submodule set")
    require(
        json.dumps(auth["required_ear_trust_vectors"], sort_keys=True) == json.dumps(vectors, sort_keys=True),
        "incorrect exact EAR trust vectors",
    )
    paths = auth["kbs_resource_paths"]
    expected_paths = {f"default/{kind}/{release}" for kind in ("image-key", "sig-public-key", "security-policy")}
    require(
        isinstance(paths, list)
        and len(paths) == 3
        and all(isinstance(p, str) for p in paths)
        and set(paths) == expected_paths,
        "expected exactly three release-scoped KBS paths",
    )
    return auth


def render_fragment(authorization, template, *, legacy=False):
    """Encode every owner-controlled value as JSON data in a local, reviewed template."""
    auth = validate_authorization(authorization)
    if legacy:
        require(auth["cpu_tee"] == "snp" and auth["gpu"] == "nvidia", "legacy fragments are SNP+GPU only")
    prefix = "wo_" + re.sub(r"[^a-z0-9_]", "_", auth["release_name"])

    def compact(value):
        return json.dumps(value, separators=(",", ":"))

    return Template(template).substitute(
        prefix=prefix,
        initdata=compact(auth["init_data_claim"]),
        image=compact(auth["encrypted_image"]),
        args=compact(auth["process_args"]),
        cpu_tee=compact(auth["cpu_tee"]),
        gpu_required=compact(auth["gpu"] == "nvidia"),
        submods=compact(auth["required_ear_submods"]),
        trust_vectors=json.dumps(auth["required_ear_trust_vectors"], indent=4),
        # Used solely to authenticate received v1 fragments. New installation
        # always renders the target-aware current template, even for legacy input.
        trust_vector=json.dumps(TRUST_VECTOR, indent=4),
        path_rules="\n".join(
            f'{prefix}_authorized_path(path) if {{ path == {compact(path.split("/"))} }}'
            for path in auth["kbs_resource_paths"]
        ),
    )


def policy_test_cases(authorization):
    """Inputs for actual OPA evaluation on secure services, not a policy emulator."""
    auth = validate_authorization(authorization)
    image, args = auth["encrypted_image"], auth["process_args"]
    cpu_tee, gpu = auth["cpu_tee"], auth["gpu"]
    evidence = {
        "init_data": auth["init_data_claim"],
        cpu_tee: (
            {"quote": {"body": {"mr_config_id": auth["init_data_claim"]}}}
            if cpu_tee == "tdx"
            else {"measurement": "approved"}
        ),
        "init_data_claims": {
            "agent_policy_claims": {
                "containers": [
                    {
                        "OCI": {
                            "Annotations": {"io.kubernetes.cri.image-name": image},
                            "Process": {"Args": args},
                        }
                    }
                ]
            }
        },
    }
    cpu = {
        "ear.trustworthiness-vector": auth["required_ear_trust_vectors"]["cpu0"],
        "ear.veraison.annotated-evidence": evidence,
        "ear.trustee.identifiers": {"validated": {"container_images": [image]}},
    }
    token = {"submods": {"cpu0": cpu}}
    if gpu == "nvidia":
        token["submods"]["gpu0"] = {
            "ear.trustworthiness-vector": auth["required_ear_trust_vectors"]["gpu0"],
            "ear.veraison.annotated-evidence": {"nvidia": {"x-nvidia-gpu-attestation-report-nonce-match": True}},
        }
    data = {"plugin": "resource", "resource-path": auth["kbs_resource_paths"][0].split("/")}
    cases = {"positive": (data, token, True), "negative-empty": ({}, {}, False)}
    for path in auth["kbs_resource_paths"]:
        entry = copy.deepcopy(data)
        entry["resource-path"] = path.split("/")
        cases["positive-" + path.split("/")[1]] = (entry, token, True)

    def change(name, parts, value, *, document="token", remove=False):
        changed_data, changed_token = copy.deepcopy(data), copy.deepcopy(token)
        target = changed_token if document == "token" else changed_data
        for part in parts[:-1]:
            target = target[part]
        if remove:
            target.pop(parts[-1])
        else:
            target[parts[-1]] = value
        cases[name] = (changed_data, changed_token, False)

    ev = ["submods", "cpu0", "ear.veraison.annotated-evidence"]
    change("negative-wrong-path", ["resource-path"], ["no", "such", "path"], document="data")
    change("negative-wrong-plugin", ["plugin"], "not-resource", document="data")
    # Guaranteed mutations, including deliberately unusual but valid owner data.
    changed_initdata = ("1" if auth["init_data_claim"][0] == "0" else "0") + auth["init_data_claim"][1:]
    change("negative-initdata", ev + ["init_data"], changed_initdata)
    change(
        "negative-unvalidated-image",
        ["submods", "cpu0", "ear.trustee.identifiers", "validated", "container_images"],
        [],
    )
    container = ev + ["init_data_claims", "agent_policy_claims", "containers", 0, "OCI"]
    change("negative-image", container + ["Annotations", "io.kubernetes.cri.image-name"], "other/image:latest")
    change("negative-command", container + ["Process", "Args"], [*args, "--unapproved-extra-argument"])
    change("negative-weak-cpu", ["submods", "cpu0", "ear.trustworthiness-vector", "hardware"], 0)
    change("negative-missing-cpu-type", ev + [cpu_tee], None, remove=True)
    change("negative-empty-cpu-type", ev + [cpu_tee], {})
    change("negative-ambiguous-cpu", ev + ["snp" if cpu_tee == "tdx" else "tdx"], {"unapproved": True})
    change("negative-unknown-cpu", ev + ["sgx"], {"unapproved": True})
    change("negative-extra-submod", ["submods", "cpu1"], cpu)
    if cpu_tee == "tdx":
        change("negative-mrconfigid", ev + ["tdx", "quote", "body", "mr_config_id"], "f" * 96)
        change("negative-truncated-initdata", ev + ["init_data"], auth["init_data_sha256"])
    if gpu == "nvidia":
        change("negative-missing-gpu", ["submods", "gpu0"], None, remove=True)
        change("negative-weak-gpu", ["submods", "gpu0", "ear.trustworthiness-vector", "hardware"], 0)
        change(
            "negative-gpu-type", ["submods", "gpu0", "ear.veraison.annotated-evidence"], {"other": {"attested": True}}
        )
    else:
        change("negative-extra-gpu", ["submods", "gpu0"], cpu)
    return cases
