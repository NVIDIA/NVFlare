#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || {
    printf 'Usage: %s /path/to/platform-reference.env\n' "$0" >&2
    exit 2
}
CONFIG_FILE="$(realpath -- "$1")"
[[ -s "${CONFIG_FILE}" ]] || { printf 'Missing configuration: %s\n' "${CONFIG_FILE}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
for command in openssl python3 realpath sha256sum; do need "${command}"; done

[[ "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'invalid PLATFORM_PROFILE'
PROFILE_DIR="${PLATFORM_WORK_ROOT:?PLATFORM_WORK_ROOT is required}/${PLATFORM_PROFILE}"
KATA_CONFIG_REL="$(cat "${PROFILE_DIR}/kata-config-relative-path.txt")"
KATA_CONFIG="${PROFILE_DIR}/${KATA_CONFIG_REL}"
[[ -s "${KATA_CONFIG}" ]] || die 'run stage 03 first'

TOOLS_ROOT="${PLATFORM_TOOLS_ROOT:-$(dirname -- "${PLATFORM_WORK_ROOT}")/platform-tools}"
VENV="${TOOLS_ROOT}/venv"
WHEEL_DIR="${TOOLS_ROOT}/wheel"
WHEEL_SHA256='feda1aebe69dcbf38e9c636ed1d082bb7dc741fc6969f197d39faddb72b2762d'
install -d -m 0700 "${TOOLS_ROOT}" "${WHEEL_DIR}"

if [[ ! -x "${VENV}/bin/sev-snp-measure" ]]; then
    python3 -m venv "${VENV}"
    "${VENV}/bin/python" -m pip download --no-deps --only-binary=:all: \
        --dest "${WHEEL_DIR}" 'sev-snp-measure==0.0.13'
    mapfile -t wheels < <(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'sev_snp_measure-0.0.13-*.whl' -print)
    (( ${#wheels[@]} == 1 )) || die 'expected exactly one sev-snp-measure 0.0.13 wheel'
    printf '%s  %s\n' "${WHEEL_SHA256}" "${wheels[0]}" | sha256sum --check --strict
    "${VENV}/bin/python" -m pip install "${wheels[0]}"
fi
SEV_TOOL="${VENV}/bin/sev-snp-measure"
[[ -x "${SEV_TOOL}" ]] || die 'sev-snp-measure installation failed'

DERIVED_ENV="${PROFILE_DIR}/platform-derived.env"
[[ ! -e "${DERIVED_ENV}" ]] || die "refusing to overwrite: ${DERIVED_ENV}"
python3 - "${KATA_CONFIG}" "${PROFILE_DIR}" "${SEV_TOOL}" > "${DERIVED_ENV}" <<'PY'
import shlex
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
profile = Path(sys.argv[2])
tool = Path(sys.argv[3])
with config_path.open("rb") as stream:
    qemu = tomllib.load(stream)["hypervisor"]["qemu"]

def relative(name, value, required=True):
    if not value:
        if required:
            raise SystemExit(f"missing qemu {name}")
        return ""
    prefix = "/opt/kata/"
    if not value.startswith(prefix):
        raise SystemExit(f"qemu {name} is outside /opt/kata: {value!r}")
    result = Path("artifacts/opt/kata") / value.removeprefix(prefix)
    if not (profile / result).is_file():
        raise SystemExit(f"derived qemu {name} does not exist: {result}")
    return str(result)

values = {
    "SEV_SNP_MEASURE": str(tool),
    "SNP_OVMF_REL": relative("firmware", qemu.get("firmware")),
    "SNP_KERNEL_REL": relative("kernel", qemu.get("kernel")),
    "SNP_INITRD_REL": relative("initrd", qemu.get("initrd"), False),
    "SNP_ROOTFS_REL": relative("image", qemu.get("image"), False),
    "SNP_KERNEL_CMDLINE_REL": "kernel-command-line.txt",
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
chmod 0600 "${DERIVED_ENV}"

KEY_DIR="${PROFILE_DIR}/authority-key"
PASS_FILE="${KEY_DIR}/platform-reference-signing.pass"
PRIVATE_KEY="${KEY_DIR}/platform-reference-signing.pem"
PUBLIC_KEY="${KEY_DIR}/platform-reference-signing.pub"
install -d -m 0700 "${KEY_DIR}"
if [[ ${PLATFORM_REFERENCE_SKIP_SIGNING:-0} != 1 ]]; then
for output in "${PASS_FILE}" "${PRIVATE_KEY}" "${PUBLIC_KEY}"; do
    [[ ! -e "${output}" ]] || die "refusing to overwrite signing material: ${output}"
done
openssl rand -hex 32 > "${PASS_FILE}"
openssl genpkey -algorithm ED25519 -aes-256-cbc -pass "file:${PASS_FILE}" -out "${PRIVATE_KEY}"
openssl pkey -in "${PRIVATE_KEY}" -passin "file:${PASS_FILE}" -pubout -out "${PUBLIC_KEY}"
chmod 0600 "${PASS_FILE}" "${PRIVATE_KEY}"
chmod 0644 "${PUBLIC_KEY}"
else
    PRIVATE_KEY=''
    PUBLIC_KEY=''
    PASS_FILE=''
    printf 'No signing key generated: this run exports only trusted reference values.\n'
fi

APPROVAL_ENV="${PROFILE_DIR}/platform-approval.env"
[[ ! -e "${APPROVAL_ENV}" ]] || die "refusing to overwrite: ${APPROVAL_ENV}"
cat > "${APPROVAL_ENV}" <<EOF
# Fill every quoted blank from independently reviewed rehearsal and firmware evidence.
PLATFORM_DESCRIPTION="$(hostname -f 2>/dev/null || hostname)"
REHEARSAL_EVIDENCE_FILE=""
TCB_EVIDENCE_FILE=""
ACCEPTED_ADVISORIES=""
APPROVED_SNP_LAUNCH_MEASUREMENT=""
SNP_MIN_REPORTED_TCB_BOOTLOADER=""
SNP_MIN_REPORTED_TCB_TEE=""
SNP_MIN_REPORTED_TCB_SNP=""
SNP_MIN_REPORTED_TCB_MICROCODE=""
EOF
chmod 0600 "${APPROVAL_ENV}"

cat >> "${DERIVED_ENV}" <<EOF
PLATFORM_REFERENCE_SIGNING_KEY='${PRIVATE_KEY}'
PLATFORM_REFERENCE_SIGNING_PUB='${PUBLIC_KEY}'
PLATFORM_REFERENCE_SIGNING_PASSPHRASE_FILE='${PASS_FILE}'
EOF

printf 'Prepared trusted platform inputs:\n  %s\n  %s\n' "${DERIVED_ENV}" "${APPROVAL_ENV}"
printf 'Next: run stage 07 to collect a challenge-bound SNP report and record its TCB floors.\n'
