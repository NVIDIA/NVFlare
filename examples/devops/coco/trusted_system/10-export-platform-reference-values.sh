#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# == 2 || $# == 3 ]] || { echo "Usage: $0 platform-reference.final.env SERVICE-VALUES.json [ADMIN-LAUNCH-PROFILE.json]" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=$(realpath -e -- "$1")
[[ $(basename -- "$CONFIG") == platform-reference.final.env ]] || {
    echo 'Use the final environment emitted by stage 09.' >&2; exit 1;
}
# Only source the trusted local stage-09 configuration, never a received handoff.
source "$CONFIG"
[[ "$CONFIG" == "${PLATFORM_WORK_ROOT}/${PLATFORM_PROFILE}/platform-reference.final.env" ]] || {
    echo 'Final environment is outside its declared platform profile.' >&2; exit 1;
}
ADMIN_PAYLOAD=''
if [[ $# == 3 ]]; then
    ADMIN_PAYLOAD="$(python3 "${SCRIPT_DIR}/export-workload-launch-profile.py" \
        "${PLATFORM_WORK_ROOT}/${PLATFORM_PROFILE}" "$PLATFORM_PROFILE" "$KATA_VERSION" \
        "$RUNTIME_CLASS" "$KATA_DEPLOY_AMD64" \
        "${APPROVED_WORKLOAD_PROFILE_SHA256:-}" "${APPROVED_ACTUAL_LAUNCH_SHA256:-}")"
fi
python3 - "$2" "${APPROVED_SNP_LAUNCH_MEASUREMENT:?Run stage 09 first}" \
    "${SNP_MIN_REPORTED_TCB_BOOTLOADER:?}" "${SNP_MIN_REPORTED_TCB_TEE:?}" \
    "${SNP_MIN_REPORTED_TCB_SNP:?}" "${SNP_MIN_REPORTED_TCB_MICROCODE:?}" \
    "${3:-}" "$ADMIN_PAYLOAD" <<'PY'
import json
from pathlib import Path
import re
import sys

output, measurement = sys.argv[1:3]
floors = sys.argv[3:7]
admin_output, admin_payload = sys.argv[7:9]
if not re.fullmatch(r"[0-9a-f]{96}", measurement):
    raise SystemExit("Invalid approved SNP launch measurement")
if any(not re.fullmatch(r"0|[1-9][0-9]{0,2}", x) or not 0 <= int(x) <= 255 for x in floors):
    raise SystemExit("All four TCB floors must be explicit decimal uint8 values")
values = {"snp_launch_measurement": measurement}
for name, value in zip(("bootloader", "tee", "snp", "microcode"), floors):
    values[f"snp_min_reported_tcb_{name}"] = int(value)
outputs = [(Path(output), json.dumps(values, indent=2) + '\n')]
if admin_output:
    json.loads(admin_payload)
    outputs.append((Path(admin_output), admin_payload.rstrip('\n') + '\n'))
if len({str(p.resolve()) for p, _ in outputs}) != len(outputs):
    raise SystemExit('Secure-services and admin outputs must be separate files')
for path, _ in outputs:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise SystemExit('Output exists or its parent directory is missing: ' + str(path))
# Exclusive creation; rollback only files created by this invocation on failure.
created = []
try:
    for path, content in outputs:
        with path.open('x') as stream:
            created.append(path)
            stream.write(content)
except BaseException:
    for path in created:
        path.unlink()
    raise
print(f"Exported exactly five platform-reference values to {output}")
if admin_output:
    print(f"Exported separate approved workload launch profile to {admin_output}")
PY
