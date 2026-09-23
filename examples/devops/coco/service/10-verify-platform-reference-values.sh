#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 platform-reference-values.json" >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
PARSER="$SCRIPT_DIR/lib/platform-reference-values.py"
python3 "$PARSER" validate "$VALUES" >/dev/null
source "$SCRIPT_DIR/lib/common.sh"
require_sudo
need_file "$KBS_CLIENT"
need_file "$ADMIN_TOKEN"
need_file "$TRUSTEE_PUBLIC_CERT"
for key in snp_launch_measurement snp_min_reported_tcb_bootloader \
    snp_min_reported_tcb_tee snp_min_reported_tcb_snp snp_min_reported_tcb_microcode; do
    raw=$(kbs_admin get-reference-value --id "$key")
    python3 "$PARSER" compare-reference "$VALUES" "$key" "$raw"
done
printf 'All five live RVPS references match the received file. No settings changed.\n'
