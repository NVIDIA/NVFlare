#!/usr/bin/env python3
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

"""Validate an authority-pinned launch contract before generation and handoff."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_profile(path, expected_sha256, runtime, kata_version):
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)), "missing/invalid authority profile SHA-256 pin")
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "launch profile SHA-256 mismatch")
    profile = json.loads(raw, object_pairs_hook=unique_object)
    keys = {
        "schema",
        "profile_id",
        "runtime_class",
        "kata_version",
        "kata_deploy_image",
        "kata_config_sha256",
        "launch_inputs_sha256",
        "vm_defaults",
        "pod_constraints",
    }
    require(isinstance(profile, dict) and set(profile) == keys, "unexpected launch-profile schema fields")
    require(profile["schema"] == "coco-approved-workload-launch/v1", "unsupported launch-profile schema")
    require(
        isinstance(profile["profile_id"], str)
        and bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", profile["profile_id"])),
        "invalid profile ID",
    )
    require(
        profile["runtime_class"] == runtime == "kata-qemu-nvidia-gpu-snp", "runtime class differs from approved profile"
    )
    require(profile["kata_version"] == kata_version == "3.29.0", "Kata version differs from approved profile")
    require(
        isinstance(profile["kata_deploy_image"], str)
        and bool(re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", profile["kata_deploy_image"])),
        "runtime image is not digest-pinned",
    )
    for key in ("kata_config_sha256", "launch_inputs_sha256"):
        require(isinstance(profile[key], str) and bool(re.fullmatch(r"[0-9a-f]{64}", profile[key])), f"invalid {key}")
    defaults = profile["vm_defaults"]
    require(isinstance(defaults, dict) and set(defaults) == {"vcpus", "memory_mib"}, "invalid VM defaults")
    require(
        all(type(value) is int and value > 0 for value in defaults.values()), "VM defaults must be positive integers"
    )
    expected = {
        "container_count": 1,
        "gpu_resource": "nvidia.com/pgpu",
        "gpu_count": 1,
        "cpu_memory_resources": "omitted",
        "host_namespaces": False,
        "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
    }
    # Canonical JSON distinguishes booleans from integers, unlike Python equality.
    require(
        json.dumps(profile["pod_constraints"], sort_keys=True) == json.dumps(expected, sort_keys=True),
        "unsupported Pod constraints; a new profile requires implementation and rehearsal review",
    )
    return profile


def validate_pod(profile, pod):
    require(isinstance(pod, dict) and pod.get("apiVersion") == "v1" and pod.get("kind") == "Pod", "expected a v1 Pod")
    spec = pod["spec"]
    allowed_spec = {
        "runtimeClassName",
        "restartPolicy",
        "hostNetwork",
        "hostPID",
        "hostIPC",
        "containers",
        "automountServiceAccountToken",
        "enableServiceLinks",
    }
    require(
        set(spec) <= allowed_spec,
        "unapproved Pod spec field (init/ephemeral containers, volumes, overhead or scheduling override)",
    )
    require(spec.get("runtimeClassName") == profile["runtime_class"], "Pod runtime class mismatch")
    for key in ("hostNetwork", "hostPID", "hostIPC"):
        require(spec.get(key, False) is False, f"{key} must be false")
    annotations = pod.get("metadata", {}).get("annotations", {})
    require(
        isinstance(annotations, dict) and set(annotations) <= set(profile["pod_constraints"]["allowed_annotations"]),
        "unapproved Pod annotation",
    )
    containers = spec.get("containers", [])
    require(isinstance(containers, list) and len(containers) == 1, "approved profile requires exactly one container")
    c = containers[0]
    allowed_container = {
        "name",
        "image",
        "imagePullPolicy",
        "command",
        "env",
        "stdin",
        "tty",
        "securityContext",
        "resources",
    }
    require(set(c) <= allowed_container, "unapproved container field")
    resources = c.get("resources", {})
    require(
        isinstance(resources, dict) and set(resources) <= {"requests", "limits"}, "unapproved resource configuration"
    )
    limits = resources.get("limits", {})
    require(
        isinstance(limits, dict) and set(limits) == {"nvidia.com/pgpu"},
        "only the approved GPU limit is allowed; CPU/memory must remain omitted",
    )

    def one(value):
        return (type(value) is int and value == 1) or (type(value) is str and value == "1")

    require(one(limits["nvidia.com/pgpu"]), "exactly one passthrough GPU is required")
    requests = resources.get("requests", {})
    require(
        isinstance(requests, dict)
        and (not requests or (set(requests) == {"nvidia.com/pgpu"} and one(requests["nvidia.com/pgpu"]))),
        "CPU/memory requests must remain omitted; GPU request must equal its limit",
    )
    require(c.get("securityContext", {}).get("privileged", False) is False, "workload container must not be privileged")


def read_pod(path):
    if Path(path).suffix == ".json":
        return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        loader.flatten_mapping(node)
        return unique_object((loader.construct_object(k), loader.construct_object(v)) for k, v in node.value)

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(Path(path).read_text(), Loader=UniqueLoader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("sha256")
    parser.add_argument("runtime")
    parser.add_argument("kata_version")
    parser.add_argument("--pod")
    parser.add_argument("--snapshot", help="exclusively create an immutable release copy")
    args = parser.parse_args()
    profile = load_profile(args.profile, args.sha256, args.runtime, args.kata_version)
    if args.pod:
        validate_pod(profile, read_pod(args.pod))
    if args.snapshot:
        raw = Path(args.profile).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == args.sha256, "profile changed before snapshot")
        with Path(args.snapshot).open("xb") as output:
            output.write(raw)
    print("Approved launch profile verified:", profile["profile_id"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit(f"Launch-profile validation failed: {error}")
