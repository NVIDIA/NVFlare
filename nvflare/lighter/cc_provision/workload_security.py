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

import base64
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from pathlib import Path

HANDOFF_FILES = {
    "cosign.pub",
    "image-security-policy.json",
    "image_key",
    "release-authorization.json",
    "resource-policy-fragment.rego",
}


def authenticate_handoff(source, destination, expected_manifest_sha256):
    """Snapshot and authenticate a confidential bundle against an out-of-band pin."""
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)), "invalid manifest pin")
    source, destination = Path(source), Path(destination)
    require(not source.is_symlink() and source.is_dir(), "handoff must be a regular directory")
    require({p.name for p in source.iterdir()} == HANDOFF_FILES | {"SHA256SUMS"}, "unexpected handoff files")
    contents = {}
    for name in sorted(HANDOFF_FILES | {"SHA256SUMS"}):
        with os.fdopen(os.open(source / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode), "handoff contains a non-regular file")
            if name == "image_key":
                require(stat.S_IMODE(info.st_mode) == 0o600, "image_key must be mode 0600")
            contents[name] = stream.read(2 * 1024 * 1024 + 1)
            require(len(contents[name]) <= 2 * 1024 * 1024, "handoff file is too large")
    manifest = contents["SHA256SUMS"]
    require(hashlib.sha256(manifest).hexdigest() == expected_manifest_sha256, "owner manifest pin mismatch")
    entries = {}
    for line in manifest.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", line)
        require(match is not None, "invalid manifest entry")
        digest, name = match.groups()
        require(name not in entries, "duplicate manifest entry")
        entries[name] = digest
    require(set(entries) == HANDOFF_FILES, "manifest must cover exactly all five payload files")
    for name, digest in entries.items():
        require(hashlib.sha256(contents[name]).hexdigest() == digest, f"payload digest mismatch: {name}")
    require(len(contents["image_key"]) == 32, "image_key must contain 32 bytes")
    destination.mkdir(mode=0o700)
    for name, content in contents.items():
        with open(destination / name, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
            output.write(content)


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
# Kata 3.29 rules with the reviewed AdditionalGids and CVE-2026-77176 derivation.
# Hash covers the whole executable preamble (only trailing whitespace normalized).
APPROVED_RULES_SHA256 = "408bec1a5a744ed5c4ceef7c5c06ed1d9c808030311e2aff72e24e1fe8cd0481"
PAUSE_IMAGE = "mcr.microsoft.com/oss/kubernetes/pause:3.6"
PAUSE_DEFAULT_CAPS = [
    "CAP_CHOWN",
    "CAP_DAC_OVERRIDE",
    "CAP_FSETID",
    "CAP_FOWNER",
    "CAP_MKNOD",
    "CAP_NET_RAW",
    "CAP_SETGID",
    "CAP_SETUID",
    "CAP_SETFCAP",
    "CAP_SETPCAP",
    "CAP_NET_BIND_SERVICE",
    "CAP_SYS_CHROOT",
    "CAP_KILL",
    "CAP_AUDIT_WRITE",
]
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
    require(
        hashlib.sha256((rules.rstrip() + "\n").encode()).hexdigest() == APPROVED_RULES_SHA256,
        "guest rule guards differ from the reviewed complete rules preamble",
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
    require(len(data["containers"]) == 2, "expected exactly one application and one pinned pause container")
    pause = next(container["OCI"] for container in data["containers"] if container["OCI"] is not matches[0])
    validate_pause_policy(pause, data)
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


def validate_pause_policy(oci, data):
    annotations = oci.get("Annotations", {})
    require(
        annotations.get("io.kubernetes.cri.container-type") == "sandbox"
        and annotations.get("io.katacontainers.pkg.oci.container_type") == "pod_sandbox",
        "unexpected non-application container",
    )
    require(annotations.get("io.kubernetes.cri.image-name", PAUSE_IMAGE) == PAUSE_IMAGE, "unapproved pause image")
    require(
        data.get("cluster_config", {}).get("pause_container_image") == PAUSE_IMAGE, "unapproved pause image setting"
    )
    process = oci.get("Process", {})
    require(process.get("Args") == ["/pause"] and process.get("NoNewPrivileges") is True, "unsafe pause process")
    user = process.get("User", {})
    require(
        type(user.get("UID")) is int and user["UID"] == 65535 and type(user.get("GID")) is int and user["GID"] == 65535,
        "pause UID/GID must match the pinned non-root profile",
    )
    require(user.get("AdditionalGids", []) == [], "unexpected pause supplementary groups")
    require(oci.get("Root", {}).get("Readonly") is True, "pause rootfs must be read-only")
    require(data.get("common", {}).get("default_caps") == PAUSE_DEFAULT_CAPS, "unapproved default capability expansion")
    caps = process.get("Capabilities", {})
    require(set(caps) == set(CAPABILITY_SETS), "invalid pause capability fields")
    for name in CAPABILITY_SETS:
        require(
            caps[name] == ([] if name in ("Ambient", "Inheritable") else ["$(default_caps)"]),
            "unapproved pause capabilities",
        )


def normalize_resources(resources):
    resources = {kind: dict(values) for kind, values in resources.items()}
    for name, value in resources.get("limits", {}).items():
        resources.setdefault("requests", {}).setdefault(name, value)
    return {kind: {name: str(value) for name, value in quantities.items()} for kind, quantities in resources.items()}


def validate_actual_launch(profile, actual, workload_path):
    """Bind captured launch inputs to the independently approved source profile."""
    require(
        hashlib.sha256(Path(workload_path).read_bytes()).hexdigest() == profile["workload_yaml_sha256"],
        "Workload source differs from the approved profile",
    )
    validate_pod_context(read_pod(workload_path), profile.get("workload_security_context"))
    require(
        actual["pod_resources"] == [profile["pod_resources"]], "Actual Pod resources differ from the approved profile"
    )
    launched = actual["artifacts"]
    require(
        launched["kata_config"]["sha256"] == profile["kata_config_sha256"],
        "Actual Kata configuration differs from the approved profile",
    )
    require("initrd" in launched or "image" in launched, "Missing actual initrd or rootfs image")
    for key in ("path", "firmware", "kernel", "initrd", "image"):
        artifact = profile["artifacts"].get(key)
        observed = launched.get("qemu_executable" if key == "path" else key)
        if key in ("initrd", "image") and observed is None:
            continue
        require(
            artifact and observed and observed["sha256"] == artifact["sha256"],
            f"Launch artifact differs from approved profile: {key}",
        )


def decode_initdata(encoded):
    require(isinstance(encoded, str) and len(encoded) <= 4 * 1024 * 1024, "invalid InitData size")
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(encoded, validate=True))) as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    require(len(raw) <= 4 * 1024 * 1024, "InitData exceeds decompressed limit")
    try:
        import tomllib
    except ImportError:
        # CoCo runs on the documented Ubuntu host (system Python 3.11+).
        # A Python 3.10 NVFlare venv can use its parser without a new package.
        parsed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-c",
                "import json,sys,tomllib; print(json.dumps(tomllib.loads(sys.stdin.read())))",
            ],
            input=raw.decode(),
            text=True,
            capture_output=True,
            timeout=10,
        )
        require(parsed.returncode == 0, "CoCo InitData validation requires system Python 3.11+")
        data = json.loads(parsed.stdout, object_pairs_hook=unique_object)
    else:
        data = tomllib.loads(raw.decode())
    return data


def validate_workload_pod(pod, expected_context, command):
    require(isinstance(pod, dict) and set(pod) <= {"apiVersion", "kind", "metadata", "spec"}, "unexpected Pod fields")
    require(pod.get("apiVersion") == "v1" and pod.get("kind") == "Pod", "expected a v1 Pod")
    spec = pod.get("spec", {})
    require(
        isinstance(spec, dict)
        and set(spec)
        <= {
            "runtimeClassName",
            "restartPolicy",
            "hostNetwork",
            "hostPID",
            "hostIPC",
            "containers",
            "automountServiceAccountToken",
            "enableServiceLinks",
        },
        "unapproved Pod spec field",
    )
    require(spec.get("runtimeClassName") == "kata-qemu-nvidia-gpu-snp", "unapproved runtime")
    for name in ("hostNetwork", "hostPID", "hostIPC"):
        require(spec.get(name, False) is False, "host namespaces are forbidden")
    require(
        spec.get("automountServiceAccountToken") is False and spec.get("enableServiceLinks") is False,
        "service account tokens and automatic service links must be disabled",
    )
    require(spec.get("restartPolicy") == "Never", "unexpected restart policy")
    validate_pod_context(pod, expected_context)
    container = spec["containers"][0]
    require(
        set(container)
        <= {"name", "image", "imagePullPolicy", "command", "env", "stdin", "tty", "securityContext", "resources"},
        "unapproved container field",
    )
    require(container.get("stdin") is False and container.get("tty") is False, "interactive streams are forbidden")
    require(
        container.get("command") == command and container.get("imagePullPolicy") == "Always",
        "unexpected workload command or pull policy",
    )
    resources = normalize_resources(container.get("resources", {}))
    require(
        resources == {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}, "unapproved resources"
    )
    metadata = pod.get("metadata", {})
    require(
        isinstance(metadata, dict) and set(metadata) <= {"name", "namespace", "labels", "annotations"},
        "unapproved Pod metadata",
    )
    annotation = "io.katacontainers.config.hypervisor.cc_init_data"
    annotations = metadata.get("annotations", {})
    require(
        isinstance(annotations, dict) and set(annotations) == {annotation},
        "unapproved annotations or missing InitData",
    )
    policy = decode_initdata(annotations[annotation])["data"]["policy.rego"]
    validate_policy(policy, container["image"], expected_context)
    _, data = validate_request_policy(policy)
    app = next(
        c["OCI"]
        for c in data["containers"]
        if c["OCI"].get("Annotations", {}).get("io.kubernetes.cri.image-name") == container["image"]
    )
    require(app["Process"].get("Args") == command, "guest command does not match the Pod")
