#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || { printf 'Usage: %s WORKLOAD.env\n' "$0" >&2; exit 2; }
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

need_cmd python3
need_file "${WORKLOAD_LAUNCH_PROFILE}"
python3 "${WORKLOAD_PROFILE_VALIDATOR}" "${WORKLOAD_LAUNCH_PROFILE}" \
    "${WORKLOAD_LAUNCH_PROFILE_SHA256:-}" "$RUNTIME_CLASS" "$KATA_VERSION"
need_file "${GENPOLICY}"
need_file "${RULES}"
need_file "${SETTINGS}"
need_file "${OUTPUT_DIR}/encrypted-image-reference.txt"
need_file "${OUTPUT_DIR}/cosign.pub"
need_file "${IMAGE_KEY_PATH}"
need_file "${REGISTRY_AUTH_FILE}"

POLICY_FILES=(
    approved-workload-launch-profile.json
    rules.rego
    genpolicy-settings.json
    image-security-policy.json
    base-initdata.toml
    pod.yaml
    final-initdata.toml
    generated-policy.rego
    expected-initdata-sha256.hex
    resource-policy-fragment.rego
    release-authorization.json
    kata-policy-security-fix.txt
    policy-SHA256SUMS
)
for file in "${POLICY_FILES[@]}"; do
    [[ ! -e "${OUTPUT_DIR}/${file}" ]] \
        || die "policy output already exists (${file}); this release is immutable"
done

POLICY_WORK_DIR="$(mktemp -d "${RELEASE_DIR}/.policy-build.XXXXXX")"
cleanup() {
    rm -rf -- "${POLICY_WORK_DIR}"
}
trap cleanup EXIT

python3 "${WORKLOAD_PROFILE_VALIDATOR}" "${WORKLOAD_LAUNCH_PROFILE}" \
    "$WORKLOAD_LAUNCH_PROFILE_SHA256" "$RUNTIME_CLASS" "$KATA_VERSION" \
    --snapshot "${POLICY_WORK_DIR}/approved-workload-launch-profile.json"
check_launch_profile() {
    python3 "${WORKLOAD_PROFILE_VALIDATOR}" \
        "${POLICY_WORK_DIR}/approved-workload-launch-profile.json" \
        "$WORKLOAD_LAUNCH_PROFILE_SHA256" "$RUNTIME_CLASS" "$KATA_VERSION" \
        --pod "${POLICY_WORK_DIR}/pod.yaml"
}

IMAGE_REF="$(tr -d '\r\n' < "${OUTPUT_DIR}/encrypted-image-reference.txt")"
[[ "${IMAGE_REF}" =~ @sha256:[0-9a-f]{64}$ ]] || die "invalid immutable image reference"

RELEASE_RULES="${POLICY_WORK_DIR}/rules.rego"
RELEASE_SETTINGS="${POLICY_WORK_DIR}/genpolicy-settings.json"
python3 - "${SETTINGS}" "${RELEASE_SETTINGS}" "${RULES}" "${RELEASE_RULES}" <<'PY'
import json
from pathlib import Path
import sys

settings = json.loads(Path(sys.argv[1]).read_text())
old = settings["kata_config"]["oci_version"]
if old != "1.1.0":
    raise SystemExit(f"unexpected packaged OCI version: {old}")
settings["kata_config"]["oci_version"] = "1.3.0"
env_regex = settings["request_defaults"]["CreateContainerRequest"]["allow_env_regex"]
if not env_regex:
    raise SystemExit("packaged allow_env_regex unexpectedly empty")
settings["request_defaults"]["CreateContainerRequest"]["allow_env_regex"] = []
Path(sys.argv[2]).write_text(json.dumps(settings, indent=4) + "\n")

rules = Path(sys.argv[3]).read_text()
old_rule = "{e | some e in p_user.AdditionalGids} == {e | some e in i_user.AdditionalGids}"
new_rule = "count({e | some e in i_user.AdditionalGids} - {e | some e in p_user.AdditionalGids}) == 0"
if rules.count(old_rule) != 1:
    raise SystemExit("unexpected AdditionalGids rule; refusing to edit")
rules = rules.replace(old_rule, new_rule, 1)

# Kata advisory GHSA-fmg6-v47x-52wr / CVE-2026-77176 affects genpolicy
# rules through Kata 4.0.0. Apply the exact semantic changes from upstream
# commit 788e2df4f4a332b2675317363ba1d2803c591d8b to this pinned 3.29 file.
old_mount_1 = '''    print("allow_mount 1: p_mount =", p_mount)
    check_mount(p_mount, i_mount, bundle_id, sandbox_id)'''
new_mount_1 = '''    print("allow_mount 1: p_mount =", p_mount)
    # check_mount expects a regex in the source field, other p_mounts are not eligible for this rule.
    p_mount.source != ""
    check_mount(p_mount, i_mount, bundle_id, sandbox_id)'''
old_mount_2 = '''    p_mount.destination == i_mount.destination
    p_mount.type_ == i_mount.type_
    p_mount.options == i_mount.options

    some i_storage in i_storages'''
new_mount_2 = '''    p_mount.destination == i_mount.destination
    p_mount.type_ == i_mount.type_
    p_mount.options == i_mount.options
    # This rule is exclusively for block-based emptyDir mounts, which don't have a source.
    p_mount.source == ""

    some i_storage in i_storages'''
old_storage = '''    some i_storage in i_storages
    print("allow_mount 2: i_storage =", i_storage)

    i_storage.mount_point == i_mount.source'''
new_storage = '''    some i_storage in i_storages
    print("allow_mount 2: i_storage =", i_storage)

    # Only block storage is a legitimate mount source for this rule.
    i_storage.driver in {"blk", "scsi"}
    i_storage.mount_point == i_mount.source'''
old_guest_pull = '''allow_storage(p_storages, i_storage, bundle_id, sandbox_id) if {
    i_storage.driver == "image_guest_pull"
    print("allow_storage with image_guest_pull: start")
    i_storage.fstype == "overlay"
    i_storage.fs_group == null
    i_storage.shared == false
    count(i_storage.options) == 0
    # TODO: Check Mount Point, Source, Driver Options, etc.
    print("allow_storage with image_guest_pull: true")
}'''
new_guest_pull = '''allow_storage(p_storages, i_storage, bundle_id, sandbox_id) if {
    print("allow_storage with image_guest_pull: start")
    i_storage.driver == "image_guest_pull"
    i_storage.fstype == "overlay"
    i_storage.fs_group == null
    i_storage.shared == false
    count(i_storage.options) == 0

    # image_guest_pull storages always target the rootfs path directly.
    expect_root_path := replace(policy_data.common.root_path, "$(bundle-id)", bundle_id)
    print("allow_storage with image_guest_pull: expect_root_path =", expect_root_path)
    expect_root_path == i_storage.mount_point

    # TODO: missing validation for fields:
    #   - driver_options
    #   - source
    print("allow_storage with image_guest_pull: true")
}'''
for name, old, new in (
    ("allow_mount nonempty source", old_mount_1, new_mount_1),
    ("allow_mount empty source", old_mount_2, new_mount_2),
    ("allow_mount block driver", old_storage, new_storage),
    ("image_guest_pull root path", old_guest_pull, new_guest_pull),
):
    if rules.count(old) != 1:
        raise SystemExit(f"unexpected upstream rules for {name}; refusing to edit")
    rules = rules.replace(old, new, 1)

Path(sys.argv[4]).write_text(rules)
PY

cat > "${POLICY_WORK_DIR}/kata-policy-security-fix.txt" <<'EOF'
Kata advisory: GHSA-fmg6-v47x-52wr
CVE: CVE-2026-77176
Affected packaged rules: Kata Containers 3.29.0
Workaround commit: 788e2df4f4a332b2675317363ba1d2803c591d8b
The policy generator applies and verifies the upstream workaround semantics.
EOF

python3 - "${POLICY_WORK_DIR}/image-security-policy.json" "${IMAGE_REPOSITORY}" \
    "$(kbs_uri "${KBS_SIGNING_KEY_PATH}")" <<'PY'
import json
from pathlib import Path
import sys

policy = {
    "default": [{"type": "reject"}],
    "transports": {
        "docker": {
            sys.argv[2]: [{
                "type": "sigstoreSigned",
                "keyPath": sys.argv[3],
                "signedIdentity": {"type": "matchRepository"},
            }]
        }
    },
}
Path(sys.argv[1]).write_text(json.dumps(policy, indent=2) + "\n")
PY

TRUSTEE_CERT="$(<"${PUBLIC_DIR}/trustee.crt")"
REGISTRY_CA="$(<"${PUBLIC_DIR}/registry-ca.crt")"
cat > "${POLICY_WORK_DIR}/base-initdata.toml" <<EOF
algorithm = "sha256"
version = "0.1.0"

[data]
"aa.toml" = '''
[token_configs]

[token_configs.kbs]
url = "${KBS_URL}"
cert = """
${TRUSTEE_CERT}
"""
'''

"cdh.toml" = '''
socket = "unix:///run/confidential-containers/cdh.sock"
credentials = []

[kbc]
name = "cc_kbc"
url = "${KBS_URL}"
kbs_cert = """
${TRUSTEE_CERT}
"""

[image]
image_security_policy_uri = "$(kbs_uri "${KBS_IMAGE_POLICY_PATH}")"
extra_root_certificates = ["""
${REGISTRY_CA}
"""]
'''
EOF
unset TRUSTEE_CERT REGISTRY_CA

python3 - "${POLICY_WORK_DIR}/pod.yaml" "${RELEASE_NAME}" "${RUNTIME_CLASS}" \
    "${IMAGE_REF}" "${APP_COMMAND_JSON}" "${APP_UID}" "${APP_GID}" \
    "${KUBERNETES_SERVICE_HOST}" "${KUBERNETES_SERVICE_PORT}" "${APP_READ_ONLY_ROOT_FILESYSTEM}" <<'PY'
import json
from pathlib import Path
import sys
import yaml

path, name, runtime, image, command_json, uid, gid, api_host, api_port, read_only = sys.argv[1:]
if read_only not in ("true", "false"):
    raise SystemExit("invalid read-only root setting")
command = json.loads(command_json)
env = [
    {"name": "KUBERNETES_PORT_443_TCP_PROTO", "value": "tcp"},
    {"name": "KUBERNETES_PORT_443_TCP_PORT", "value": api_port},
    {"name": "KUBERNETES_PORT_443_TCP_ADDR", "value": api_host},
    {"name": "KUBERNETES_SERVICE_HOST", "value": api_host},
    {"name": "KUBERNETES_SERVICE_PORT", "value": api_port},
    {"name": "KUBERNETES_SERVICE_PORT_HTTPS", "value": api_port},
    {"name": "KUBERNETES_PORT", "value": f"tcp://{api_host}:{api_port}"},
    {"name": "KUBERNETES_PORT_443_TCP", "value": f"tcp://{api_host}:{api_port}"},
]
pod = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {"name": name, "namespace": "default", "labels": {"app": name}},
    "spec": {
        "runtimeClassName": runtime,
        "restartPolicy": "Never",
        "hostNetwork": False,
        "containers": [{
            "name": name,
            "image": image,
            "imagePullPolicy": "Always",
            "command": command,
            "env": env,
            "stdin": False,
            "tty": False,
            "securityContext": {
                "privileged": False,
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": read_only == "true",
                "runAsUser": int(uid),
                "runAsGroup": int(gid),
                "capabilities": {"drop": ["ALL"]},
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "resources": {"limits": {"nvidia.com/pgpu": "1"}},
        }],
    },
}
Path(path).write_text(yaml.safe_dump(pod, sort_keys=False))
PY

export DOCKER_CONFIG="${REGISTRY_SECRET_DIR}"
check_launch_profile
"${GENPOLICY}" \
    --rego-rules-path "${RELEASE_RULES}" \
    --json-settings-path "${RELEASE_SETTINGS}" \
    --initdata-path="${POLICY_WORK_DIR}/base-initdata.toml" \
    --yaml-file "${POLICY_WORK_DIR}/pod.yaml"

python3 - "${POLICY_WORK_DIR}" "${IMAGE_REF}" "${APP_COMMAND_JSON}" \
    "${RUNTIME_CLASS}" "${APP_UID}" "${APP_GID}" "${APP_READ_ONLY_ROOT_FILESYSTEM}" <<'PY'
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
import yaml

out = Path(sys.argv[1])
image = sys.argv[2]
command = json.loads(sys.argv[3])
runtime = sys.argv[4]
uid, gid = int(sys.argv[5]), int(sys.argv[6])
pod_path = out / "pod.yaml"
pod = yaml.safe_load(pod_path.read_text())

# Kata 3.29 strict genpolicy rejects these Kubernetes-only fields. Add them
# after generation; they do not broaden the generated guest OCI policy.
pod["spec"].update({
    "automountServiceAccountToken": False,
    "enableServiceLinks": False,
    "hostPID": False,
    "hostIPC": False,
})
pod["spec"]["containers"][0]["securityContext"]["runAsNonRoot"] = True
pod_path.write_text(yaml.safe_dump(pod, sort_keys=False))

annotation = pod["metadata"]["annotations"][
    "io.katacontainers.config.hypervisor.cc_init_data"
]
raw = gzip.decompress(base64.b64decode(annotation, validate=True))
(out / "final-initdata.toml").write_bytes(raw)
data = tomllib.loads(raw.decode("utf-8"))
policy = data["data"]["policy.rego"]
(out / "generated-policy.rego").write_text(policy)
digest_hex = hashlib.sha256(raw).hexdigest()
(out / "expected-initdata-sha256.hex").write_text(digest_hex + "\n")

c = pod["spec"]["containers"][0]
assert c["image"] == image and c["command"] == command
assert pod["spec"]["runtimeClassName"] == runtime
assert c["stdin"] is False and c["tty"] is False
assert c["securityContext"]["runAsUser"] == uid
assert c["securityContext"]["runAsGroup"] == gid
assert c["securityContext"]["privileged"] is False
assert c["securityContext"]["allowPrivilegeEscalation"] is False
assert c["securityContext"]["readOnlyRootFilesystem"] is (sys.argv[7] == "true")
assert pod["spec"]["automountServiceAccountToken"] is False
assert "volumes" not in pod["spec"] and "volumeMounts" not in c
assert "ports" not in c and "envFrom" not in c and "args" not in c

for request in ("ExecProcessRequest", "ReadStreamRequest", "WriteStreamRequest", "SetPolicyRequest"):
    if not re.search(rf"default\s+{request}\s*:?=\s*false", policy):
        raise SystemExit(f"generated policy does not default-deny {request}")
if not re.search(r"AllowRequestsFailingPolicy[^\n]*false", policy):
    raise SystemExit("generated policy does not fail closed")
for expected in [image, *command]:
    if json.dumps(expected) not in policy and expected not in policy:
        raise SystemExit(f"generated policy lacks exact value: {expected}")
if re.search(r'"exec_commands"\s*:\s*\[\s*[^\]]', policy):
    raise SystemExit("generated policy unexpectedly authorizes an exec command")
for expected_fix in (
    'p_mount.source != ""',
    'p_mount.source == ""',
    'i_storage.driver in {"blk", "scsi"}',
    'expect_root_path == i_storage.mount_point',
):
    if expected_fix not in policy:
        raise SystemExit(f"generated policy lacks CVE-2026-77176 workaround: {expected_fix}")
print("Pod and generated agent-policy invariants verified")
PY

check_launch_profile
EXPECTED_INITDATA_HEX="$(tr -d '\r\n' < "${POLICY_WORK_DIR}/expected-initdata-sha256.hex")"
[[ "${EXPECTED_INITDATA_HEX}" =~ ^[0-9a-f]{64}$ ]] \
    || die "invalid lowercase-hex SNP init-data digest"

python3 - "${POLICY_WORK_DIR}" "${RELEASE_NAME}" "${IMAGE_REF}" \
    "${APP_COMMAND_JSON}" "${EXPECTED_INITDATA_HEX}" \
    "${KBS_IMAGE_KEY_PATH}" "${KBS_SIGNING_KEY_PATH}" "${KBS_IMAGE_POLICY_PATH}" <<'PY'
import json
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
release, image = sys.argv[2], sys.argv[3]
args = json.loads(sys.argv[4])
initdata = sys.argv[5]
paths = sys.argv[6:9]
prefix = "wo_" + re.sub(r"[^a-z0-9_]", "_", release)

def q(value):
    return json.dumps(value, separators=(",", ":"))

path_rules = "\n".join(
    f'{prefix}_authorized_path(path) if {{ path == {q(path.split("/"))} }}'
    for path in paths
)
fragment = f'''# Merge this fragment into the trusted service administrator's global
# package policy. Do not add another package/import/default declaration.

{prefix}_expected_initdata := {q(initdata)}
{prefix}_expected_image := {q(image)}
{prefix}_expected_args := {q(args)}
{prefix}_expected_trust_vector := {{
    "executables": 3,
    "hardware": 2,
    "configuration": 3,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}}

{path_rules}

{prefix}_approved_trust_vector(submod) if {{
    submod["ear.trustworthiness-vector"] == {prefix}_expected_trust_vector
}}

{prefix}_approved_container(container) if {{
    container["OCI"]["Annotations"]["io.kubernetes.cri.image-name"] == {prefix}_expected_image
    container["OCI"]["Process"]["Args"] == {prefix}_expected_args
}}

allow if {{
    data.plugin == "resource"
    {prefix}_authorized_path(data["resource-path"])
    count(input.submods) == 2
    {prefix}_approved_trust_vector(input.submods.cpu0)
    {prefix}_approved_trust_vector(input.submods.gpu0)

    cpu := input.submods.cpu0
    cpu["ear.veraison.annotated-evidence"]["init_data"] == {prefix}_expected_initdata
    {prefix}_expected_image in cpu["ear.trustee.identifiers"]["validated"]["container_images"]

    containers := cpu["ear.veraison.annotated-evidence"]["init_data_claims"]["agent_policy_claims"]["containers"]
    some container in containers
    {prefix}_approved_container(container)
}}
'''
(out / "resource-policy-fragment.rego").write_text(fragment)

authorization = {
    "schema": "coco-workload-owner-authorization/v1",
    "release_name": release,
    "encrypted_image": image,
    "process_args": args,
    "snp_init_data_sha256": initdata,
    "snp_init_data_encoding_in_trustee_v0_21": "lowercase-hex",
    "required_ear_submods": ["cpu0", "gpu0"],
    "required_ear_trust_vectors": {
        name: {
            "executables": 3,
            "hardware": 2,
            "configuration": 3,
            "file-system": 0,
            "instance-identity": 0,
            "runtime-opaque": 0,
            "storage-opaque": 0,
            "sourced-data": 0,
        }
        for name in ("cpu0", "gpu0")
    },
    "kbs_resource_paths": paths,
}
(out / "release-authorization.json").write_text(json.dumps(authorization, indent=2) + "\n")
PY

(
    cd "${POLICY_WORK_DIR}"
    sha256sum approved-workload-launch-profile.json pod.yaml final-initdata.toml generated-policy.rego \
        image-security-policy.json release-authorization.json \
        resource-policy-fragment.rego kata-policy-security-fix.txt \
        > policy-SHA256SUMS
)

for file in "${POLICY_FILES[@]}"; do
    install -m 0600 "${POLICY_WORK_DIR}/${file}" "${OUTPUT_DIR}/${file}"
done

printf 'Generated the measured Pod and trusted-service authorization inputs under %s.\n' "${OUTPUT_DIR}"
printf 'Review them before creating either handoff.\n'
