#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

for command in containerd helm kubectl; do need_cmd "${command}"; done
kctl get nodes
kctl get runtimeclass "${RUNTIME_CLASS}"
kctl get pods -A -o wide
kctl get node -o jsonpath='{range .items[*]}{.metadata.name}{" pgpu="}{.status.allocatable.nvidia\.com/pgpu}{"\n"}{end}'
helmctl list -A
containerd --version
kubectl version --client
printf 'Registry trust files:\n'
sudo find "/etc/containerd/certs.d/${REGISTRY_HOST}" -maxdepth 1 -type f \
    -printf '  %f %m %u:%g\n' | sort
printf 'CoCo platform verification completed. Review all displayed state.\n'
