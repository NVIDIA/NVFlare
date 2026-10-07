#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
selected_runtime="$RUNTIME_CLASS"
source "$SCRIPT_DIR/bootstrap/lib/common.sh"
load_config
[[ "$RUNTIME_CLASS" == "$selected_runtime" ]] || die 'platform.env and config.env select different runtimes'

for command in containerd helm kubectl python3; do need_cmd "${command}"; done
config_name="$(python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" target "$RUNTIME_CLASS" --field config_name)"
gpu_count="$(python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" target "$RUNTIME_CLASS" --field gpu_count)"
kctl get nodes
kctl get runtimeclass "${RUNTIME_CLASS}"
kctl get pods -A -o wide
if ((gpu_count > 0)); then
    kctl get node -o jsonpath='{range .items[*]}{.metadata.name}{" pgpu="}{.status.allocatable.nvidia\.com/pgpu}{"\n"}{end}'
fi
helmctl list -A
containerd --version
runtime_config="/opt/kata/share/defaults/kata-containers/$config_name"
as_root python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" check-target "$RUNTIME_CLASS" "$runtime_config"
as_root python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" check \
    "$runtime_config" \
    --runtime /opt/kata/bin/kata-runtime
validate_runtime_prerequisites "$RUNTIME_CLASS" "$runtime_config"
kubectl version --client
printf 'Registry trust files:\n'
as_root find "/etc/containerd/certs.d/${REGISTRY_HOST}" -maxdepth 1 -type f \
    -printf '  %f %m %u:%g\n' | sort
printf 'CoCo platform verification completed. Review all displayed state.\n'
