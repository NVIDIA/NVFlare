#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 PLATFORM_ENV" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(realpath -e "$1")"
source "$SCRIPT_DIR/lib/validate-config.sh"
validate_target_host
(( EUID != 0 )) || config_error 'Run as the intended unprivileged operator with sudo.'
sudo -n true
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl ca-certificates jq python3-venv python3-yaml skopeo openssl zstd pciutils
sudo systemctl enable --now docker
sudo usermod -aG docker "$(id -un)"
printf 'Host tools installed. Open a new SSH session for Docker group membership.\n'
