#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 2 ]] || {
    printf 'Usage: %s BASE-CONFIG APPROVAL-ENV\n' "$0" >&2
    exit 2
}
BASE_CONFIG="$(realpath -- "$1")"
APPROVAL_ENV="$(realpath -- "$2")"
[[ -s "${BASE_CONFIG}" && -s "${APPROVAL_ENV}" ]] || { printf 'Missing input file\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "${BASE_CONFIG}"
PROFILE_DIR="${PLATFORM_WORK_ROOT:?PLATFORM_WORK_ROOT is required}/${PLATFORM_PROFILE}"
DERIVED_ENV="${PROFILE_DIR}/platform-derived.env"
[[ -s "${DERIVED_ENV}" ]] || { printf 'Run stage 06 first\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "${DERIVED_ENV}"
# shellcheck source=/dev/null
source "${APPROVAL_ENV}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for command in cmp python3 sha256sum; do
    command -v "${command}" >/dev/null 2>&1 || die "missing command: ${command}"
done
for variable in PLATFORM_DESCRIPTION REHEARSAL_EVIDENCE_FILE \
    TCB_EVIDENCE_FILE \
    SNP_MIN_REPORTED_TCB_BOOTLOADER SNP_MIN_REPORTED_TCB_TEE \
    SNP_MIN_REPORTED_TCB_SNP SNP_MIN_REPORTED_TCB_MICROCODE; do
    [[ -n "${!variable-}" ]] || die "approval field is blank: ${variable}"
done

KATA_CONFIG_REL="$(cat "${PROFILE_DIR}/kata-config-relative-path.txt")"
PINNED_KATA_CONFIG="${PROFILE_DIR}/${KATA_CONFIG_REL}"
INSTALLED_KATA_CONFIG='/opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml'
KATA_RUNTIME='/opt/kata/bin/kata-runtime'
[[ -s "${PINNED_KATA_CONFIG}" && -s "${INSTALLED_KATA_CONFIG}" && -x "${KATA_RUNTIME}" ]] \
    || die 'pinned or installed Kata SNP runtime input is missing'
mapfile -d '' -t AUTO_LAUNCH_INPUTS < <(python3 - "${PINNED_KATA_CONFIG}" "${KATA_RUNTIME}" \
    "${INSTALLED_KATA_CONFIG}" <<'PY'
import json
import subprocess
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    pinned_config = tomllib.load(stream)
with open(sys.argv[3], "rb") as stream:
    installed_config = tomllib.load(stream)
if installed_config != pinned_config:
    raise SystemExit("installed Kata SNP settings differ from the pinned artifact")
qemu = pinned_config["hypervisor"]["qemu"]
vcpus = qemu.get("default_vcpus")
if not isinstance(vcpus, int) or vcpus < 1:
    raise SystemExit("pinned default_vcpus is not a positive integer")

runtime_env = json.loads(subprocess.check_output(
    [sys.argv[2], "--config", sys.argv[3], "kata-env", "--json"], text=True
))
cmdline = runtime_env.get("Kernel", {}).get("Parameters")
if not isinstance(cmdline, str) or not cmdline:
    raise SystemExit("Kata did not report its effective kernel parameters")
configured = qemu.get("kernel_params", "").split()
effective = cmdline.split()
if any(item not in effective for item in configured):
    raise SystemExit("effective Kata kernel parameters omit a pinned configured parameter")

cpu = {}
with open("/proc/cpuinfo", encoding="utf-8") as stream:
    for line in stream:
        if not line.strip() and cpu:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            cpu[key.strip()] = value.strip()
if cpu.get("vendor_id") != "AuthenticAMD":
    raise SystemExit("trusted rehearsal host is not an AMD CPU")
family = int(cpu["cpu family"])
model = int(cpu["model"])
stepping = int(cpu["stepping"])
if family < 0x0f or model > 0xff or stepping > 0x0f:
    raise SystemExit("CPU identity cannot be encoded as an x86 CPUID signature")
signature = ((stepping & 0xf) | ((model & 0xf) << 4) | (0xf << 8)
             | (((model >> 4) & 0xf) << 16) | ((family - 0xf) << 20))

values = [str(vcpus), "", hex(signature), "0x1", cmdline]
sys.stdout.write("\0".join(values) + "\0")
PY
)
(( ${#AUTO_LAUNCH_INPUTS[@]} == 5 )) || die 'failed to extract five Kata launch inputs'
SNP_VCPUS="${AUTO_LAUNCH_INPUTS[0]}"
SNP_VCPU_TYPE="${AUTO_LAUNCH_INPUTS[1]}"
SNP_VCPU_SIG="${AUTO_LAUNCH_INPUTS[2]}"
SNP_GUEST_FEATURES="${AUTO_LAUNCH_INPUTS[3]}"
SNP_KERNEL_CMDLINE="${AUTO_LAUNCH_INPUTS[4]}"
is_uint8() { [[ "$1" =~ ^(0|[1-9][0-9]{0,2})$ ]] && ((10#$1 <= 255)); }
for variable in SNP_MIN_REPORTED_TCB_BOOTLOADER SNP_MIN_REPORTED_TCB_TEE \
    SNP_MIN_REPORTED_TCB_SNP SNP_MIN_REPORTED_TCB_MICROCODE; do
    is_uint8 "${!variable}" || die "${variable} must be a decimal uint8"
done

safe_evidence() {
    local name="$1" value="$2"
    [[ "${value}" != /* && "${value}" != .. && "${value}" != ../* && "${value}" != */../* ]] \
        || die "${name} must be relative to the profile directory"
    [[ -s "${PROFILE_DIR}/${value}" ]] || die "missing ${name}: ${PROFILE_DIR}/${value}"
}
safe_evidence REHEARSAL_EVIDENCE_FILE "${REHEARSAL_EVIDENCE_FILE}"
safe_evidence TCB_EVIDENCE_FILE "${TCB_EVIDENCE_FILE}"

# Stage 07 retains the complete signed report and AMD certificate chain beside
# the human-readable TCB evidence. Re-verify them here, then extract the launch
# measurement directly from the signed SNP report. This is the authoritative
# value: the offline launch model below is retained only as a diagnostic.
TCB_EVIDENCE_DIR="$(dirname -- "${PROFILE_DIR}/${TCB_EVIDENCE_FILE}")"
SIGNED_REPORT="${TCB_EVIDENCE_DIR}/attestation-report.bin"
REPORT_CERTS="${TCB_EVIDENCE_DIR}/certs"
REHEARSAL_RUN="${PROFILE_DIR}/rehearsal-collector-build"
if [[ -n "${REHEARSAL_WORKLOAD_YAML:-}" ]]; then
    [[ -s "${REHEARSAL_RUN}/actual-launch.json" && -s "${PROFILE_DIR}/approved-launch-profile.json" ]] || die 'Missing approved profile or actual launch capture'
    for binding in \
        "Actual launch SHA-256|$(sha256sum "${REHEARSAL_RUN}/actual-launch.json" | awk '{print $1}')" \
        "Workload source SHA-256|$(sha256sum "${REHEARSAL_WORKLOAD_YAML}" | awk '{print $1}')"; do
        field="${binding%%|*}"
        [[ $(sed -n "s/^${field}: //p" "${PROFILE_DIR}/${REHEARSAL_EVIDENCE_FILE}") == "${binding#*|}" ]] || die "Changed ${field}"
    done
    python3 - "${PROFILE_DIR}/approved-launch-profile.json" "${REHEARSAL_RUN}/actual-launch.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
profile, actual = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
if actual['pod_resources'] != [profile['pod_resources']]:
    raise SystemExit('Actual Pod resources differ from the approved workload profile')
if actual['artifacts']['kata_config']['sha256'] != profile['kata_config_sha256']:
    raise SystemExit('Actual Kata configuration differs from the approved profile')
for key, artifact in profile['artifacts'].items():
    target = 'qemu_executable' if key == 'path' else 'configured_' + key
    if actual['artifacts'][target]['sha256'] != artifact['sha256']:
        raise SystemExit(f'Launch artifact differs from approved profile: {key}')
PY
    APPROVED_WORKLOAD_PROFILE_SHA256="$(sha256sum "${PROFILE_DIR}/approved-launch-profile.json" | awk '{print $1}')"
    APPROVED_ACTUAL_LAUNCH_SHA256="$(sha256sum "${REHEARSAL_RUN}/actual-launch.json" | awk '{print $1}')"
fi
SNP_GUEST="${REHEARSAL_RUN}/snpguest"
[[ -s "${SIGNED_REPORT}" && -d "${REPORT_CERTS}" ]] \
    || die 'stage-07 signed report or AMD certificate chain is missing'
[[ -x "${SNP_GUEST}" ]] || die 'retained checksum-pinned stage-07 snpguest is missing'
mapfile -t SNP_GUEST_HASHES < <(
    sed -n 's/^snpguest SHA-256: //p' "${PROFILE_DIR}/${REHEARSAL_EVIDENCE_FILE}"
)
(( ${#SNP_GUEST_HASHES[@]} == 1 )) \
    || die 'rehearsal evidence must contain exactly one snpguest SHA-256'
[[ "$(sha256sum "${SNP_GUEST}" | awk '{print $1}')" == "${SNP_GUEST_HASHES[0]}" ]] \
    || die 'retained snpguest differs from the rehearsal evidence pin'
(
    cd "${TCB_EVIDENCE_DIR}"
    sha256sum --check --strict SHA256SUMS
)
"${SNP_GUEST}" verify certs "${REPORT_CERTS}"
"${SNP_GUEST}" verify attestation "${REPORT_CERTS}" "${SIGNED_REPORT}"
"${SNP_GUEST}" verify attestation "${REPORT_CERTS}" "${SIGNED_REPORT}" --tcb

REHEARSAL_INPUT="${REHEARSAL_RUN}/evidence-input"
[[ -s "${REHEARSAL_RUN}/request-data.bin" && -s "${REHEARSAL_INPUT}/request-data.bin" \
    && -s "${REHEARSAL_INPUT}/attestation-report.bin" ]] \
    || die 'retained stage-07 challenge or attestation evidence is missing'
(
    cd "${REHEARSAL_INPUT}"
    sha256sum --check --strict SHA256SUMS
)
cmp -- "${REHEARSAL_RUN}/request-data.bin" "${REHEARSAL_INPUT}/request-data.bin" \
    || die 'retained collector challenge differs from the verified evidence challenge'
cmp -- "${SIGNED_REPORT}" "${REHEARSAL_INPUT}/attestation-report.bin" \
    || die 'TCB report differs from the challenge-bound rehearsal report'

read -r APPROVED_SNP_LAUNCH_MEASUREMENT REPORT_BOOTLOADER REPORT_TEE REPORT_SNP \
    REPORT_MICROCODE < <(python3 - "${SIGNED_REPORT}" "${REHEARSAL_RUN}/request-data.bin" <<'PY'
import hmac
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_bytes()
challenge = Path(sys.argv[2]).read_bytes()
if len(report) < 0x188:
    raise SystemExit("attestation report is truncated")
if len(challenge) != 64 or not hmac.compare_digest(report[0x50:0x90], challenge):
    raise SystemExit("signed REPORT_DATA does not match the retained fresh challenge")
tcb = report[0x180:0x188]
if any(tcb[2:6]):
    raise SystemExit("reserved REPORTED_TCB bytes are nonzero")
print(report[0x90:0xc0].hex(), tcb[0], tcb[1], tcb[6], tcb[7])
PY
)
[[ "${APPROVED_SNP_LAUNCH_MEASUREMENT}" =~ ^[0-9a-f]{96}$ ]] \
    || die 'signed report contains an invalid SNP launch measurement'
for binding in \
    "Reported launch measurement|${APPROVED_SNP_LAUNCH_MEASUREMENT}" \
    "Challenge SHA-256|$(sha256sum "${REHEARSAL_RUN}/request-data.bin" | awk '{print $1}')" \
    "Attestation report SHA-256|$(sha256sum "${SIGNED_REPORT}" | awk '{print $1}')"; do
    field="${binding%%|*}"
    expected="${binding#*|}"
    mapfile -t values < <(sed -n "s/^${field}: //p" \
        "${PROFILE_DIR}/${REHEARSAL_EVIDENCE_FILE}")
    (( ${#values[@]} == 1 )) || die "rehearsal evidence must contain exactly one ${field}"
    [[ "${values[0]}" == "${expected}" ]] || die "rehearsal evidence ${field} is inconsistent"
done
[[ "${REPORT_BOOTLOADER}" == "${SNP_MIN_REPORTED_TCB_BOOTLOADER}" \
    && "${REPORT_TEE}" == "${SNP_MIN_REPORTED_TCB_TEE}" \
    && "${REPORT_SNP}" == "${SNP_MIN_REPORTED_TCB_SNP}" \
    && "${REPORT_MICROCODE}" == "${SNP_MIN_REPORTED_TCB_MICROCODE}" ]] \
    || die 'approved minimum TCB values differ from the verified signed report'

CMDLINE_FILE="${PROFILE_DIR}/${SNP_KERNEL_CMDLINE_REL}"
printf '%s' "${SNP_KERNEL_CMDLINE}" > "${CMDLINE_FILE}"
chmod 0600 "${CMDLINE_FILE}"

args=(--mode snp --vmm-type QEMU --vcpus "${SNP_VCPUS}" \
    --guest-features "${SNP_GUEST_FEATURES}" --ovmf "${PROFILE_DIR}/${SNP_OVMF_REL}" \
    --kernel "${PROFILE_DIR}/${SNP_KERNEL_REL}" --append "${SNP_KERNEL_CMDLINE}" \
    --output-format hex)
if [[ -n "${SNP_VCPU_TYPE-}" ]]; then args+=(--vcpu-type "${SNP_VCPU_TYPE}"); else args+=(--vcpu-sig "${SNP_VCPU_SIG}"); fi
[[ -z "${SNP_INITRD_REL-}" ]] || args+=(--initrd "${PROFILE_DIR}/${SNP_INITRD_REL}")
MODELED_SNP_LAUNCH_MEASUREMENT="$("${SEV_SNP_MEASURE}" "${args[@]}")"
[[ "${MODELED_SNP_LAUNCH_MEASUREMENT}" =~ ^[0-9a-f]{96}$ ]] \
    || die 'measurement calculator returned an invalid SNP launch measurement'

TCB_APPROVAL_FILE='tcb-approval.txt'
cat > "${PROFILE_DIR}/${TCB_APPROVAL_FILE}" <<EOF
Platform: ${PLATFORM_DESCRIPTION}
Rehearsal evidence: ${REHEARSAL_EVIDENCE_FILE}
TCB evidence: ${TCB_EVIDENCE_FILE}
Accepted advisories: ${ACCEPTED_ADVISORIES:-none}
Approved SNP launch measurement: ${APPROVED_SNP_LAUNCH_MEASUREMENT}
Measurement source: cryptographically verified stage-07 SNP attestation report
Offline modeled SNP launch measurement (diagnostic only): ${MODELED_SNP_LAUNCH_MEASUREMENT}
Minimum reported TCB bootloader: ${SNP_MIN_REPORTED_TCB_BOOTLOADER}
Minimum reported TCB TEE: ${SNP_MIN_REPORTED_TCB_TEE}
Minimum reported TCB SNP: ${SNP_MIN_REPORTED_TCB_SNP}
Minimum reported TCB microcode: ${SNP_MIN_REPORTED_TCB_MICROCODE}
EOF
chmod 0600 "${PROFILE_DIR}/${TCB_APPROVAL_FILE}"

FINAL_ENV="${PROFILE_DIR}/platform-reference.final.env"
[[ ! -e "${FINAL_ENV}" ]] || die "refusing to overwrite: ${FINAL_ENV}"
{
    printf '# Generated by stage 09 from reviewed inputs.\n'
    for variable in PLATFORM_PROFILE PLATFORM_WORK_ROOT KATA_VERSION RUNTIME_CLASS KATA_CHART_OCI \
        KATA_CHART_OCI_DIGEST KATA_CHART_TGZ_SHA256 KATA_DEPLOY_INDEX KATA_DEPLOY_AMD64 \
        SEV_SNP_MEASURE SNP_OVMF_REL SNP_KERNEL_REL SNP_INITRD_REL SNP_ROOTFS_REL \
        SNP_KERNEL_CMDLINE_REL SNP_VCPUS SNP_VCPU_TYPE SNP_VCPU_SIG SNP_GUEST_FEATURES \
        APPROVED_SNP_LAUNCH_MEASUREMENT MODELED_SNP_LAUNCH_MEASUREMENT \
        APPROVED_WORKLOAD_PROFILE_SHA256 APPROVED_ACTUAL_LAUNCH_SHA256 \
        TCB_APPROVAL_FILE REHEARSAL_EVIDENCE_FILE TCB_EVIDENCE_FILE \
        SNP_MIN_REPORTED_TCB_BOOTLOADER SNP_MIN_REPORTED_TCB_TEE \
        SNP_MIN_REPORTED_TCB_SNP SNP_MIN_REPORTED_TCB_MICROCODE PLATFORM_REFERENCE_SIGNING_KEY \
        PLATFORM_REFERENCE_SIGNING_PUB PLATFORM_REFERENCE_SIGNING_PASSPHRASE_FILE; do
        printf '%s=%q\n' "${variable}" "${!variable-}"
    done
} > "${FINAL_ENV}"
chmod 0600 "${FINAL_ENV}"
printf 'Final platform configuration created: %s\nVerified and approved measurement: %s\n' \
    "${FINAL_ENV}" "${APPROVED_SNP_LAUNCH_MEASUREMENT}"
printf 'Offline modeled measurement (diagnostic only): %s\n' \
    "${MODELED_SNP_LAUNCH_MEASUREMENT}"
