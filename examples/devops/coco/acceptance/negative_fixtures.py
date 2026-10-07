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

"""Prepare private, unapplied Pod candidates for reviewed operator denial tests."""

import argparse
import base64
import copy
import gzip
import hashlib
import io
import json
import os
import re
import tomllib
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

INITDATA = "io.katacontainers.config.hypervisor.cc_init_data"
MAX_INITDATA_BYTES = 4 * 1024 * 1024


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key, deep=deep)
        if key in result:
            raise ValueError("Duplicate YAML field")
        result[key] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def initdata_bytes(pod):
    annotation = pod["metadata"]["annotations"][INITDATA]
    compressed = base64.b64decode(annotation, validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(MAX_INITDATA_BYTES + 1)
    if len(raw) > MAX_INITDATA_BYTES:
        raise ValueError("InitData exceeds fixture size limit")
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("algorithm") != "sha256" or not isinstance(data.get("data", {}).get("policy.rego"), str):
        raise ValueError("Measured InitData with SHA256 and embedded policy is required")
    return raw


def identity_audit(pod):
    spec = pod["spec"]
    container = spec["containers"][0]
    effective = dict(spec.get("securityContext", {}))
    effective.update(container.get("securityContext", {}))
    return {
        "pod_name": pod["metadata"]["name"],
        "namespace": pod["metadata"].get("namespace", "default"),
        "pod_uid": pod["metadata"].get("uid"),
        "run_as_user": effective.get("runAsUser"),
        "run_as_group": effective.get("runAsGroup"),
        "run_as_non_root": effective.get("runAsNonRoot"),
        "supplemental_groups": effective.get("supplementalGroups"),
        "fs_group": effective.get("fsGroup"),
    }


def baseline_pod(raw):
    pod = yaml.load(raw, Loader=UniqueLoader)
    if not isinstance(pod, dict) or pod.get("apiVersion") != "v1" or pod.get("kind") != "Pod":
        raise ValueError("A single v1 Pod is required")
    if not isinstance(pod.get("metadata"), dict) or not pod["metadata"].get("name"):
        raise ValueError("Pod identity is required")
    spec = pod.get("spec", {})
    if spec.get("runtimeClassName") != "kata-qemu-tdx" or len(spec.get("containers", [])) != 1:
        raise ValueError("A single protected kata-qemu-tdx application container is required")
    if spec.get("initContainers") or spec.get("ephemeralContainers") or spec.get("volumes"):
        raise ValueError("Baseline must contain no added containers or volumes")
    if spec.get("hostNetwork", False) or spec.get("hostPID", False) or spec.get("hostIPC", False):
        raise ValueError("Baseline must not expose host namespaces")
    container = spec["containers"][0]
    if container.get("volumeMounts") or not re.fullmatch(r".+@sha256:[a-f0-9]{64}", container.get("image", "")):
        raise ValueError("A digest-pinned baseline without mounts is required")
    if not container.get("command") or container.get("command") == ["/bin/true"]:
        raise ValueError("An explicit reviewed application command is required")
    audit = identity_audit(pod)
    if audit["run_as_user"] != 65532 or audit["run_as_group"] != 65532 or audit["run_as_non_root"] is not True:
        raise ValueError("Baseline must use approved UID/GID65532 and non-root execution")
    context = container.get("securityContext", {})
    if context.get("privileged") is not False or context.get("allowPrivilegeEscalation") is not False:
        raise ValueError("Baseline must prohibit privilege escalation")
    initdata_bytes(pod)
    return pod


def differences(before, after, path=""):
    """Audit changed JSON-pointer paths without echoing potentially sensitive values."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            pointer = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                result.append(pointer)
            else:
                result.extend(differences(before[key], after[key], pointer))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [p for index, (a, b) in enumerate(zip(before, after)) for p in differences(a, b, f"{path}/{index}")]
    return [] if before == after else [path]


def prepare(pod_path, expected_sha256, output, run_id, topology, participant, namespace, authority_note):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", run_id) or run_id.endswith("-"):
        raise ValueError("Run ID must be a DNS label of at most 24 characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", namespace) or namespace.endswith("-"):
        raise ValueError("An isolated DNS-label namespace is required")
    if topology not in ("A", "B") or participant not in ("site-1", "site-2", "server"):
        raise ValueError("Invalid topology or participant")
    if participant == "server" and topology != "B":
        raise ValueError("Topology A server is not protected")
    if not authority_note.strip():
        raise ValueError("Authenticated signed-handoff digest provenance is required")
    raw = Path(pod_path).read_bytes()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or sha256(raw) != expected_sha256:
        raise ValueError("Baseline differs from authenticated signed-handoff digest")
    baseline = baseline_pod(raw)
    if namespace == baseline["metadata"].get("namespace", "default"):
        raise ValueError("Fixture namespace must differ from baseline")
    control = copy.deepcopy(baseline)
    control.pop("status", None)
    for key in ("uid", "resourceVersion", "generation", "creationTimestamp", "managedFields", "ownerReferences"):
        control["metadata"].pop(key, None)
    control["metadata"]["namespace"] = namespace
    control["metadata"]["name"] = f"negative-{run_id}-control"
    variants = {"control": control}
    for name in ("changed-command", "weakened-context", "host-mount", "altered-initdata"):
        variants[name] = copy.deepcopy(control)
        variants[name]["metadata"]["name"] = f"negative-{run_id}-{name}"
    variants["changed-command"]["spec"]["containers"][0]["command"] = ["/bin/true"]
    variants["changed-command"]["spec"]["containers"][0]["args"] = []
    variants["weakened-context"]["spec"]["containers"][0]["securityContext"].update(
        runAsUser=0, runAsGroup=0, runAsNonRoot=False
    )
    variants["host-mount"]["spec"]["volumes"] = [
        {"name": "acceptance-host-proc", "hostPath": {"path": "/proc", "type": "Directory"}}
    ]
    variants["host-mount"]["spec"]["containers"][0]["volumeMounts"] = [
        {"name": "acceptance-host-proc", "mountPath": "/acceptance-host-proc", "readOnly": True}
    ]
    # Preserve valid UTF-8 TOML and all embedded policy strings. A comment changes
    # the measured raw InitData hash used by the provisioned release contract.
    altered = initdata_bytes(control) + f"\n# acceptance-negative-fixture: {run_id}\n".encode()
    variants["altered-initdata"]["metadata"]["annotations"][INITDATA] = base64.b64encode(
        gzip.compress(altered, mtime=0)
    ).decode("ascii")
    expectations = {
        "control": "Establish fresh isolated control guest and successful nonce job before attributing denials.",
        "changed-command": "Expect CreateContainer/guest agent policy denial of unapproved command.",
        "weakened-context": "Expect CreateContainer/guest OCI policy denial of UID/GID/context drift.",
        "host-mount": "Expect runtime/guest policy denial of unapproved host-backed volume/process access.",
        "altered-initdata": "Expect appraisal/resource authorization denial for changed measured InitData/MRCONFIGID.",
    }
    manifest = {
        "schema": "nvflare-tdx-negative-fixtures/v1",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "topology": topology,
        "participant": participant,
        "namespace": namespace,
        "baseline": {
            "sha256": expected_sha256,
            "identity": identity_audit(baseline),
            "authority_note": authority_note,
            "initdata_raw_sha256": sha256(initdata_bytes(baseline)),
        },
        "control_differences_from_baseline": differences(baseline, control),
        "applied": False,
        "approval_changed": False,
        "policy_or_resource_written": False,
        "hardware_acceptance": "NOT_RUN",
        "fixtures": {},
        "operator_requirements": [
            "Review candidates and confirm authorized isolated namespace, runtime, and untrusted-role RBAC.",
            "No bypass of trusted launcher or newly broadened approvals is provided by this preparer.",
            "If launcher/RBAC/network denies first, record that layer; it does not satisfy guest/KBS denial coverage.",
            "Record actual Pod UID, timestamps, events, exact enforcement layer and sanitized denial receipt.",
            "Delete only the operator-created fixture Pod after observation; restore baseline and pass a fresh nonce job.",
            "Never expose guest token API responses or private participant credentials in receipts.",
            "A prepared candidate or exit code alone is not a hardware PASS.",
        ],
    }
    old_umask = os.umask(0o077)
    try:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name, pod in variants.items():
            payload = yaml.safe_dump(pod, sort_keys=False).encode()
            filename = name + ".pod.yaml"
            (output / filename).write_bytes(payload)
            manifest["fixtures"][name] = {
                "file": filename,
                "sha256": sha256(payload),
                "identity": identity_audit(pod),
                "differences_from_isolated_control": differences(control, pod),
                "initdata_raw_sha256": sha256(initdata_bytes(pod)),
                "expected_observation": expectations[name],
                "observed_status": "NOT_RUN",
            }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        os.umask(old_umask)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True, help="Reviewed baseline Pod from signed provisioning handoff")
    parser.add_argument("--baseline-sha256", required=True, help="Digest authenticated from signed handoff")
    parser.add_argument("--output", required=True, help="New private directory; existing outputs are never overwritten")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topology", required=True, choices=("A", "B"))
    parser.add_argument("--participant", required=True, choices=("site-1", "site-2", "server"))
    parser.add_argument(
        "--namespace", required=True, help="Operator-authorized isolated namespace, distinct from baseline"
    )
    parser.add_argument("--authority-note", required=True, help="Sanitized provenance of authenticated baseline digest")
    args = parser.parse_args(argv)
    try:
        prepare(
            args.pod,
            args.baseline_sha256,
            args.output,
            args.run_id,
            args.topology,
            args.participant,
            args.namespace,
            args.authority_note,
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, zlib.error) as error:
        parser.exit(1, f"Fixture preparation failed ({type(error).__name__}); inspect inputs privately.\n")
    print("Private fixture candidates prepared; no Pods applied or policies/approvals changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
