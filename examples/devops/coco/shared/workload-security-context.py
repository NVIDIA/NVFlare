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

"""Approved application security context, distinct from collector privileges.

Only the reviewed single-container Linux shape is supported. These checks run
on trusted systems; cluster-side preflight is not the security boundary.
"""

import json
import re

SCHEMA = "coco-approved-workload-launch/v3"
CONTEXT_KEYS = {
    "privileged",
    "allowPrivilegeEscalation",
    "runAsNonRoot",
    "runAsUser",
    "runAsGroup",
    "readOnlyRootFilesystem",
    "capabilities",
    "seccompProfile",
}
CAPABILITY_SETS = ("Ambient", "Bounding", "Effective", "Inheritable", "Permitted")
DENIED_REQUESTS = ("ExecProcessRequest", "ReadStreamRequest", "WriteStreamRequest", "SetPolicyRequest")
# Structural checks of the pinned rules, not a general-purpose Rego verifier.
REQUIRED_GUARDS = (
    "p_oci.Root.Readonly == i_oci.Root.Readonly",
    "p_process.NoNewPrivileges == i_process.NoNewPrivileges",
    "p_user.UID == i_user.UID",
    "p_user.GID == i_user.GID",
    "allow_caps(p_process.Capabilities, i_process.Capabilities)",
    "is_null(i_linux.Seccomp)",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate key: {key}")
        result[key] = value
    return result


def validate_context(context):
    """Return a canonical, explicitly approved context; never infer safe defaults."""
    require(isinstance(context, dict) and set(context) == CONTEXT_KEYS, "explicit securityContext fields required")
    for key, expected in (("privileged", False), ("allowPrivilegeEscalation", False), ("runAsNonRoot", True)):
        require(context[key] is expected, f"securityContext.{key} must be {expected}")
    for key in ("runAsUser", "runAsGroup"):
        require(type(context[key]) is int and 0 < context[key] < 2**32 - 1, f"{key} must be a nonzero numeric ID")
    require(type(context["readOnlyRootFilesystem"]) is bool, "readOnlyRootFilesystem must be an explicit boolean")
    caps = context["capabilities"]
    require(
        isinstance(caps, dict)
        and set(caps) <= {"drop", "add"}
        and caps.get("drop") == ["ALL"]
        and caps.get("add", []) == [],
        "drop ALL capabilities and do not add capabilities",
    )
    require(context["seccompProfile"] == {"type": "RuntimeDefault"}, "only RuntimeDefault seccomp is approved")
    result = dict(context)
    result["capabilities"] = {"drop": ["ALL"]}
    result["seccompProfile"] = {"type": "RuntimeDefault"}
    return result


def pod_context(pod):
    spec = pod["spec"]
    require("securityContext" not in spec, "Pod-level securityContext overrides are not supported")
    containers = spec["containers"]
    require(isinstance(containers, list) and len(containers) == 1, "exactly one application container required")
    return validate_context(containers[0].get("securityContext"))


def validate_pod_context(pod, expected):
    require(
        pod_context(pod) == validate_context(expected), "Pod securityContext differs from approved workload context"
    )


def read_pod(path):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        loader.flatten_mapping(node)
        return unique_object((loader.construct_object(k), loader.construct_object(v)) for k, v in node.value)

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(path.read_text(), Loader=UniqueLoader)


def validate_request_policy(policy):
    """Validate effective request settings in the pinned Kata 3.29 policy format.

    This is not an arbitrary-Rego equivalence checker: the generation workflow
    must still use reviewed rules. A constant default-deny rule alone does not
    disable a request that has a conditional allow rule.
    """
    parts = re.split(r"(?m)^policy_data := ", policy)
    require(len(parts) == 2, "expected one pinned genpolicy JSON policy_data assignment")
    rules, raw = parts
    data = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(data, dict), "policy_data must be an object")
    defaults = data.get("request_defaults")
    require(isinstance(defaults, dict), "explicit request_defaults required")
    for name in DENIED_REQUESTS:
        require(
            re.search(rf"(?m)^\s*default\s+{name}\s*:?=\s*false\s*(?:#.*)?$", rules),
            f"missing default-deny {name}",
        )
    for name in ("ReadStreamRequest", "WriteStreamRequest"):
        require(defaults.get(name) is False, f"request_defaults.{name} must be false")
    # SetPolicy has no conditional allow rule or settings entry in the pin.
    # Reject an added rule or a future setting that could enable replacement.
    require(defaults.get("SetPolicyRequest", False) is False, "SetPolicyRequest must remain disabled")
    require(
        len(re.findall(r"(?m)^\s*(?:default\s+)?SetPolicyRequest\b", rules)) == 1,
        "unexpected SetPolicyRequest rule",
    )
    fail_open = re.findall(r"(?m)^\s*(?:default\s+)?AllowRequestsFailingPolicy\b[^\n]*", rules)
    require(
        len(fail_open) == 1
        and re.fullmatch(r"\s*(?:default\s+)?AllowRequestsFailingPolicy\s*:?=\s*false\s*(?:#.*)?", fail_open[0]),
        "policy must fail closed",
    )
    exec_defaults = defaults.get("ExecProcessRequest")
    require(
        isinstance(exec_defaults, dict)
        and set(exec_defaults) == {"allowed_commands", "regex"}
        and exec_defaults["allowed_commands"] == []
        and exec_defaults["regex"] == [],
        "global exec command and regex allowlists must be empty",
    )
    containers = data.get("containers")
    require(isinstance(containers, list) and bool(containers), "container policies required")
    for container in containers:
        require(
            isinstance(container, dict) and container.get("exec_commands") == [],
            "each container exec_commands allowlist must be explicitly empty",
        )
    return rules, data


def validate_policy(policy, image, context):
    """Check actual genpolicy OCI data and pinned guard presence before publication.

    Kata 3.29 requires null guest OCI Seccomp. RuntimeDefault is a YAML
    requirement, NOT a claim that this guest enforces a seccomp filter.
    """
    expected = validate_context(context)
    rules, data = validate_request_policy(policy)
    for guard in REQUIRED_GUARDS:
        require(guard in rules, f"missing pinned guest security guard: {guard}")
    for name in CAPABILITY_SETS:
        require(f"match_caps(p_caps.{name}, i_caps.{name})" in rules, f"missing guest capability guard: {name}")
    matches = [
        container["OCI"]
        for container in data["containers"]
        if container["OCI"].get("Annotations", {}).get("io.kubernetes.cri.image-name") == image
    ]
    require(len(matches) == 1, "expected exactly one application OCI policy")
    oci = matches[0]
    process = oci["Process"]
    for field, key in (("UID", "runAsUser"), ("GID", "runAsGroup")):
        value = process["User"][field]
        require(type(value) is int and value == expected[key], f"guest policy {field} differs from approved context")
    require(process["NoNewPrivileges"] is True, "guest policy must enforce NoNewPrivileges")
    caps = process["Capabilities"]
    require(
        isinstance(caps, dict) and set(caps) == set(CAPABILITY_SETS) and all(caps[k] == [] for k in CAPABILITY_SETS),
        "guest policy must grant no Linux capabilities",
    )
    require(oci["Root"]["Readonly"] is expected["readOnlyRootFilesystem"], "guest policy rootfs mode mismatch")
