#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

need_file() {
    [[ -s "$1" ]] || die "required file is missing or empty: $1"
}

[[ $# -eq 1 ]] || die "usage: $0 TRUSTED-SERVICE-HANDOFF-DIRECTORY"

HANDOFF_DIR="$(readlink -f -- "$1")"
[[ -d "${HANDOFF_DIR}" ]] || die "handoff is not a directory: ${HANDOFF_DIR}"

KBS_CERT="${TRUSTEE_PUBLIC_CERT}"
ACTIVE_POLICY="${TRUSTEE_ROOT}/kbs/data/kbs-policy/resource-policy.rego"
KBS_STORAGE="${TRUSTEE_ROOT}/kbs/data/kbs-storage"
COMPOSE_FILE="${TRUSTEE_ROOT}/docker-compose.yml"
BACKUP_ROOT="${HOME}/trustee-release-backups"

for command in awk chmod cmp cp curl diff docker find grep install jq mktemp \
    opa openssl python3 readlink sha256sum shred sort stat sudo wc; do
    need_cmd "${command}"
done
for file in "${KBS_CLIENT}" "${KBS_CERT}" "${ADMIN_TOKEN}" \
    "${ACTIVE_POLICY}" "${COMPOSE_FILE}"; do
    need_file "${file}"
done
[[ -x "${KBS_CLIENT}" ]] || die "KBS client is not executable: ${KBS_CLIENT}"
[[ -d "${KBS_STORAGE}" ]] || die "KBS storage directory is missing: ${KBS_STORAGE}"

EXPECTED_FILES=(
    SHA256SUMS
    cosign.pub
    image-security-policy.json
    image_key
    release-authorization.json
    resource-policy-fragment.rego
)
for file in "${EXPECTED_FILES[@]}"; do
    need_file "${HANDOFF_DIR}/${file}"
done
mapfile -d '' -t RECEIVED_FILES < <(
    find "${HANDOFF_DIR}" -mindepth 1 -maxdepth 1 -type f -print0 | sort -z
)
[[ "${#RECEIVED_FILES[@]}" -eq "${#EXPECTED_FILES[@]}" ]] \
    || die "handoff must contain exactly the six expected regular files"
[[ "$(find "${HANDOFF_DIR}" -mindepth 1 -maxdepth 1 ! -type f | wc -l)" -eq 0 ]] \
    || die "handoff contains an unexpected directory or special file"

(
    cd "${HANDOFF_DIR}"
    sha256sum --check --strict SHA256SUMS
)
[[ "$(wc -c < "${HANDOFF_DIR}/image_key")" -eq 32 ]] \
    || die "image_key must contain exactly 32 bytes"
[[ "$(stat -c '%a' "${HANDOFF_DIR}/image_key")" == "600" ]] \
    || die "received image_key must be mode 0600"

WORK_DIR="$(mktemp -d -p /tmp coco-service-install.XXXXXXXX)"
trap 'rm -rf -- "${WORK_DIR}"' EXIT
REVIEWED_FRAGMENT="${WORK_DIR}/reviewed-resource-policy-fragment.rego"

mapfile -d '' -t RELEASE_META < <(
    python3 - "${HANDOFF_DIR}" "${SERVICE_FQDN}:${REGISTRY_PORT}" \
        "${SCRIPT_DIR}/policies/workload-resource-policy.rego.template" "${REVIEWED_FRAGMENT}" <<'PY'
import json
from pathlib import Path
import re
import sys
from string import Template

root = Path(sys.argv[1])
authorization = json.loads((root / "release-authorization.json").read_text())
image_policy = json.loads((root / "image-security-policy.json").read_text())
fragment = (root / "resource-policy-fragment.rego").read_text()
cosign_public = (root / "cosign.pub").read_text()

if authorization.get("schema") != "coco-workload-owner-authorization/v1":
    raise SystemExit("unsupported release-authorization schema")
release = authorization.get("release_name", "")
if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", release) or len(release) > 63:
    raise SystemExit("invalid release_name")

image = authorization.get("encrypted_image", "")
image_match = re.fullmatch(
    r"(" + re.escape(sys.argv[2]) + r"/[a-z0-9._/-]+)"
    r"@(sha256:[0-9a-f]{64})",
    image,
)
if not image_match:
    raise SystemExit("encrypted_image is not an immutable image in the approved registry")
repository = image_match.group(1)

args = authorization.get("process_args")
if not isinstance(args, list) or not args or not all(isinstance(v, str) and v for v in args):
    raise SystemExit("process_args must be a non-empty string array")
if not args[0].startswith("/"):
    raise SystemExit("the executable in process_args must be an absolute path")

initdata = authorization.get("snp_init_data_sha256", "")
if not re.fullmatch(r"[0-9a-f]{64}", initdata):
    raise SystemExit("SNP init-data value is not 64-character lowercase hex")
if authorization.get("snp_init_data_encoding_in_trustee_v0_21") != "lowercase-hex":
    raise SystemExit("unexpected SNP init-data encoding")
if authorization.get("required_ear_submods") != ["cpu0", "gpu0"]:
    raise SystemExit("authorization does not require exactly CPU and GPU submodules")
expected_vector = {
    "executables": 3,
    "hardware": 2,
    "configuration": 3,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}
required_vectors = {name: expected_vector for name in ("cpu0", "gpu0")}
if authorization.get("required_ear_trust_vectors") != required_vectors:
    raise SystemExit("authorization lacks the exact approved CPU/GPU trust vectors")

paths = authorization.get("kbs_resource_paths")
expected_paths = {
    f"default/image-key/{release}",
    f"default/sig-public-key/{release}",
    f"default/security-policy/{release}",
}
if not isinstance(paths, list) or set(paths) != expected_paths or len(paths) != 3:
    raise SystemExit("KBS paths are not the three unique release paths")
key_path = next(v for v in paths if v.startswith("default/image-key/"))
signing_path = next(v for v in paths if v.startswith("default/sig-public-key/"))
policy_path = next(v for v in paths if v.startswith("default/security-policy/"))

if image_policy.get("default") != [{"type": "reject"}]:
    raise SystemExit("image policy is not default-deny")
docker_policy = image_policy.get("transports", {}).get("docker", {})
if set(docker_policy) != {repository}:
    raise SystemExit("image policy authorizes an unexpected repository")
rules = docker_policy[repository]
expected_rule = {
    "type": "sigstoreSigned",
    "keyPath": f"kbs:///{signing_path}",
    "signedIdentity": {"type": "matchRepository"},
}
if rules != [expected_rule]:
    raise SystemExit("image policy does not contain the exact expected signature rule")
if not cosign_public.startswith("-----BEGIN PUBLIC KEY-----\n"):
    raise SystemExit("cosign.pub is not a PEM public key")

prefix = "wo_" + re.sub(r"[^a-z0-9_]", "_", release)
def compact(value):
    return json.dumps(value, separators=(",", ":"))

# Only this secure-services-owned template can contribute executable Rego.
# JSON encoding keeps owner-supplied strings as data; Template substitution
# is single-pass, so placeholder-like text in arguments is not interpreted.
rendered = Template(Path(sys.argv[3]).read_text()).substitute(
    prefix=prefix, initdata=compact(initdata), image=compact(image), args=compact(args),
    path_rules="\n".join(
        f'{prefix}_authorized_path(path) if {{ path == {compact(path.split("/"))} }}'
        for path in paths
    ),
)
if fragment.strip() != rendered.strip():
    raise SystemExit("received policy fragment differs from the secure-services template")
Path(sys.argv[4]).write_text(rendered)

values = [release, key_path, signing_path, policy_path, prefix]
sys.stdout.write("\0".join(values) + "\0")
PY
)
[[ "${#RELEASE_META[@]}" -eq 5 ]] || die "failed to obtain validated release metadata"
RELEASE_NAME="${RELEASE_META[0]}"
KEY_PATH="${RELEASE_META[1]}"
SIGNING_PATH="${RELEASE_META[2]}"
IMAGE_POLICY_PATH="${RELEASE_META[3]}"
POLICY_PREFIX="${RELEASE_META[4]}"

TOKEN_MODE="$(stat -c '%a' "${ADMIN_TOKEN}")"
TOKEN_OWNER="$(stat -c '%U:%G' "${ADMIN_TOKEN}")"
[[ "${TOKEN_MODE}" == "600" && "${TOKEN_OWNER}" == "root:root" ]] \
    || die "admin token must be root:root mode 0600; see README.md"

openssl verify -CAfile "${KBS_CERT}" "${KBS_CERT}" >/dev/null
curl --fail --silent --show-error --cacert "${KBS_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"

# Hold this lock for the entire read-merge-approve-write critical section below,
# through the final policy commit. It is shared with
# 09-install-platform-policy.sh, which touches the same active resource
# policy file. Without it, two concurrent installers (or an installer racing a
# platform-policy update) can both read the same starting policy, merge
# independently, and overwrite each other on write.
lock_platform_reference_update

if grep -Fq "${POLICY_PREFIX}_expected_initdata" "${ACTIVE_POLICY}"; then
    die "release ${RELEASE_NAME} already exists in the active policy; releases are immutable"
fi

CANDIDATE_POLICY="${WORK_DIR}/merged-resource-policy.rego"

python3 - "${ACTIVE_POLICY}" "${REVIEWED_FRAGMENT}" \
    "${CANDIDATE_POLICY}" <<'PY'
from pathlib import Path
import re
import sys

active = Path(sys.argv[1]).read_text()
fragment = Path(sys.argv[2]).read_text()
if len(re.findall(r"^package policy\s*$", active, re.MULTILINE)) != 1:
    raise SystemExit("active policy must contain exactly one 'package policy'")
if len(re.findall(r"^import rego\.v1\s*$", active, re.MULTILINE)) != 1:
    raise SystemExit("active policy must contain exactly one 'import rego.v1'")
if len(re.findall(r"^default allow\s*:=\s*false\s*$", active, re.MULTILINE)) != 1:
    raise SystemExit("active policy must contain exactly one default-deny declaration")
if re.search(r"^\s*(package|import|default)\b", fragment, re.MULTILINE):
    raise SystemExit("fragment contains a forbidden top-level declaration")
Path(sys.argv[3]).write_text(active.rstrip() + "\n\n" + fragment.lstrip())
PY

opa check --strict "${CANDIDATE_POLICY}"

# Defense-in-depth tests for the merged policy. These sampled inputs are not
# a sandbox for arbitrary Rego: the trusted template above is the boundary.
mapfile -d '' -t SEMANTIC_CASES < <(
    python3 - "${HANDOFF_DIR}/release-authorization.json" "${WORK_DIR}" <<'PY'
import json
from pathlib import Path
import sys

auth_path, work_dir = Path(sys.argv[1]), Path(sys.argv[2])
authorization = json.loads(auth_path.read_text())
image = authorization["encrypted_image"]
args = authorization["process_args"]
initdata = authorization["snp_init_data_sha256"]
trust_vector = next(iter(authorization["required_ear_trust_vectors"].values()))
paths = authorization["kbs_resource_paths"]
approved_path = sorted(paths)[0]

def submod(init_data=initdata, image_name=image, process_args=args, vector=trust_vector):
    return {
        "ear.trustworthiness-vector": vector,
        "ear.veraison.annotated-evidence": {
            "init_data": init_data,
            "init_data_claims": {
                "agent_policy_claims": {
                    "containers": [
                        {"OCI": {
                            "Annotations": {"io.kubernetes.cri.image-name": image_name},
                            "Process": {"Args": process_args},
                        }}
                    ]
                }
            },
        },
        "ear.trustee.identifiers": {"validated": {"container_images": [image_name]}},
    }

approved = submod()
wrong_vector = dict(trust_vector)
wrong_vector["hardware"] = 0

def doc(resource_path="", plugin="", cpu=None, gpu=None):
    data = {}
    if resource_path:
        # The generated {prefix}_authorized_path() rule compares
        # data["resource-path"] against path.split("/"): KBS supplies the
        # resource path as a list of segments, not a slash-joined string.
        data["resource-path"] = resource_path.split("/")
    if plugin:
        data["plugin"] = plugin
    submods = {}
    if cpu is not None:
        submods["cpu0"] = cpu
    if gpu is not None:
        submods["gpu0"] = gpu
    input_doc = {"submods": submods} if submods else {}
    return data, input_doc

cases = {
    "positive": (*doc(approved_path, "resource", approved, approved), True),
    "negative-empty": (*doc(), False),
    "negative-wrong-path": (*doc("no/such/path", "resource", approved, approved), False),
    "negative-wrong-plugin": (*doc(approved_path, "not-resource", approved, approved), False),
    "negative-tampered-initdata": (
        *doc(approved_path, "resource", submod(init_data="0" * 64), approved), False,
    ),
    "negative-tampered-image": (
        *doc(approved_path, "resource", submod(image_name="docker.io/other/image:latest"), approved),
        False,
    ),
    "negative-weak-trust-vector": (
        *doc(approved_path, "resource", submod(vector=wrong_vector), approved), False,
    ),
    "negative-missing-gpu": (*doc(approved_path, "resource", approved, None), False),
}
# count(input.submods) == 2 must reject a third submodule.
extra_data, extra_input = doc(approved_path, "resource", approved, approved)
extra_input["submods"]["cpu1"] = approved
cases["negative-extra-submod"] = (extra_data, extra_input, False)

out = []
for name, (data, input_doc, expected) in cases.items():
    data_file = work_dir / f"{name}.data.json"
    input_file = work_dir / f"{name}.input.json"
    data_file.write_text(json.dumps(data))
    input_file.write_text(json.dumps(input_doc))
    out.append(f"{name}={'true' if expected else 'false'}={data_file}={input_file}")
sys.stdout.write("\0".join(out) + "\0")
PY
)
[[ "${#SEMANTIC_CASES[@]}" -gt 0 ]] || die "failed to generate semantic policy test cases"

for case in "${SEMANTIC_CASES[@]}"; do
    IFS='=' read -r CASE_NAME CASE_EXPECTED CASE_DATA CASE_INPUT <<<"${case}"
    CASE_RESULT="$(opa eval --format raw \
        --data "${CANDIDATE_POLICY}" --data "${CASE_DATA}" --input "${CASE_INPUT}" \
        'data.policy.allow')"
    [[ "${CASE_RESULT}" == "${CASE_EXPECTED}" ]] \
        || die "merged policy failed semantic test '${CASE_NAME}': expected allow=${CASE_EXPECTED}, got ${CASE_RESULT}"
done
printf 'Merged policy passed %d semantic authorization tests (fail-closed on empty/tampered/wrong-path evidence).\n' \
    "${#SEMANTIC_CASES[@]}"

printf '\nValidated release: %s\n' "${RELEASE_NAME}"
printf 'Policy change that will be installed:\n'
DIFF_STATUS=0
diff --unified "${ACTIVE_POLICY}" "${CANDIDATE_POLICY}" || DIFF_STATUS=$?
[[ "${DIFF_STATUS}" -le 1 ]] || die "failed to display policy diff"

[[ -t 0 ]] || die "interactive terminal required for final approval"
read -r -p "Type the release name to approve this exact merge: " CONFIRM_RELEASE
[[ "${CONFIRM_RELEASE}" == "${RELEASE_NAME}" ]] || die "release approval did not match"

# Some OS images permit passwordless commands while `sudo -v` still asks for
# a password because of sudoers verifypw policy. Prove noninteractive sudo
# authorization using an actual no-op command instead.
sudo -n true
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_ROOT}/${RELEASE_NAME}-${STAMP}"
install -d -m 0700 "${BACKUP_ROOT}"
mkdir -m 0700 "${BACKUP_DIR}"
cp -a "${ACTIVE_POLICY}" "${BACKUP_DIR}/active-resource-policy-before.rego"
cp -a "${COMPOSE_FILE}" "${BACKUP_DIR}/docker-compose.yml"
cp -a "${CANDIDATE_POLICY}" "${BACKUP_DIR}/merged-resource-policy.rego"
cp -a "${HANDOFF_DIR}/release-authorization.json" \
    "${HANDOFF_DIR}/resource-policy-fragment.rego" \
    "${HANDOFF_DIR}/image-security-policy.json" \
    "${HANDOFF_DIR}/cosign.pub" "${HANDOFF_DIR}/SHA256SUMS" \
    "${BACKUP_DIR}/"
sudo cp -a "${KBS_STORAGE}" "${BACKUP_DIR}/kbs-storage-before"
sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${COMPOSE_FILE}" cp \
    kbs:/opt/confidential-containers/kbs/kbs/resource-policy.rego \
    "${BACKUP_DIR}/active-resource-policy-in-container-before.rego"
sudo chown -R "$(id -u):$(id -g)" "${BACKUP_DIR}"
chmod -R u=rwX,go= "${BACKUP_DIR}"

KBS=(
    sudo "${KBS_CLIENT}"
    --url "${KBS_URL}"
    --cert-file "${KBS_CERT}"
    config
    --admin-token-file "${ADMIN_TOKEN}"
)

# Install every release resource -- signing key, image policy, and the image
# key itself -- and confirm each one persisted correctly *before* committing
# the merged resource policy. The policy commit below is what the
# already-exists/immutable-release check at the top of this script keys off
# of, so it must be the last write: if any earlier step fails, the release is
# not yet recorded as installed and this script can simply be rerun for the
# same release. Committing the policy first (as before) let a failed key
# upload leave a release permanently stuck: it looked "already installed" to
# the immutability check, but the key had never been persisted.
"${KBS[@]}" set-resource --path "${SIGNING_PATH}" \
    --resource-file "${HANDOFF_DIR}/cosign.pub" >/dev/null
"${KBS[@]}" set-resource --path "${IMAGE_POLICY_PATH}" \
    --resource-file "${HANDOFF_DIR}/image-security-policy.json" >/dev/null
"${KBS[@]}" set-resource --path "${KEY_PATH}" \
    --resource-file "${HANDOFF_DIR}/image_key" >/dev/null
printf 'Installed the three release resources.\n'

mapfile -d '' -t STORAGE_NAMES < <(
    python3 - "${KEY_PATH}" "${SIGNING_PATH}" "${IMAGE_POLICY_PATH}" <<'PY'
import sys
sys.stdout.write("\0".join(value.replace("/", r"\x2F") for value in sys.argv[1:]) + "\0")
PY
)
INSTALLED_KEY="${KBS_STORAGE}/${STORAGE_NAMES[0]}"
INSTALLED_SIGNING_KEY="${KBS_STORAGE}/${STORAGE_NAMES[1]}"
INSTALLED_IMAGE_POLICY="${KBS_STORAGE}/${STORAGE_NAMES[2]}"
for file in "${INSTALLED_KEY}" "${INSTALLED_SIGNING_KEY}" "${INSTALLED_IMAGE_POLICY}"; do
    sudo test -s "${file}" || die "expected persisted KBS resource is missing: ${file}"
done
sudo chmod 0600 "${INSTALLED_KEY}"
sudo cmp --silent "${HANDOFF_DIR}/image_key" "${INSTALLED_KEY}" \
    || die "persisted image key differs from received key"
sudo cmp --silent "${HANDOFF_DIR}/cosign.pub" "${INSTALLED_SIGNING_KEY}" \
    || die "persisted signing key differs from received key"
sudo cmp --silent "${HANDOFF_DIR}/image-security-policy.json" "${INSTALLED_IMAGE_POLICY}" \
    || die "persisted image policy differs from received policy"
printf 'Verified all three persisted resources match the received handoff.\n'

"${KBS[@]}" set-resource-policy --policy-file "${CANDIDATE_POLICY}" >/dev/null
CANDIDATE_HASH="$(sha256sum "${CANDIDATE_POLICY}" | awk '{print $1}')"
ACTIVE_HASH="$(sha256sum "${ACTIVE_POLICY}" | awk '{print $1}')"
[[ "${ACTIVE_HASH}" == "${CANDIDATE_HASH}" ]] \
    || die "persisted resource policy does not match the reviewed candidate"
printf 'Installed and verified the complete merged default-deny resource policy.\n'

curl --fail --silent --show-error --cacert "${KBS_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"
sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${COMPOSE_FILE}" ps

python3 - "${BACKUP_DIR}/installation-receipt.txt" "${RELEASE_NAME}" \
    "${ACTIVE_HASH}" "${KEY_PATH}" "${SIGNING_PATH}" "${IMAGE_POLICY_PATH}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys

receipt = (
    f"installed_at={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    f"release_name={sys.argv[2]}\n"
    f"resource_policy_sha256={sys.argv[3]}\n"
    f"image_key_path={sys.argv[4]}\n"
    f"signing_key_path={sys.argv[5]}\n"
    f"image_policy_path={sys.argv[6]}\n"
)
Path(sys.argv[1]).write_text(receipt)
PY
chmod 0600 "${BACKUP_DIR}/installation-receipt.txt"

printf '\nRelease %s is installed and persisted resource hashes match.\n' "${RELEASE_NAME}"
printf 'Protected backup and receipt: %s\n' "${BACKUP_DIR}"
read -r -p 'Type REMOVE to shred the received staging image_key now, or press Enter to retain it: ' REMOVE_KEY
if [[ "${REMOVE_KEY}" == "REMOVE" ]]; then
    shred --remove -- "${HANDOFF_DIR}/image_key"
    printf 'Removed the received staging key. KBS storage remains authoritative.\n'
else
    chmod 0600 "${HANDOFF_DIR}/image_key"
    printf 'Staging key retained at %s; keep the directory private.\n' "${HANDOFF_DIR}/image_key"
fi
