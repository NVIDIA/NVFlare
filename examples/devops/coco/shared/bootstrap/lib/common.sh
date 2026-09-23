#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../lib" && pwd)/common-base.sh"
# Cluster-only bootstrap helpers; no secure-services deployment.
SUITE_DIR="${COCO_BOOTSTRAP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${COCO_CONFIG:-${SUITE_DIR}/config.env}"
STATE_DIR="${COCO_STATE_DIR:-${HOME}/.local/state/coco-bootstrap}"
load_config() {
  [[ -s "$CONFIG_FILE" ]] || die "Copy config.env.example to $CONFIG_FILE, chmod 600, and review it first"
  source "$CONFIG_FILE"
  source "$SUITE_DIR/../lib/validate-config.sh"
  validate_target_host
  [[ ${TEE_PLATFORM:-} == snp && ${RUNTIME_CLASS:-} == kata-qemu-nvidia-gpu-snp ]] || die "This package supports AMD SNP plus NVIDIA confidential GPU only"
  [[ ${IGNORE_CHECKSUM_MISMATCH:-0} == 0 ]] || die "Checksum bypass is not supported"
  IGNORE_CHECKSUM_MISMATCH=0
  [[ ${ALLOW_PACKAGE_DOWNGRADES:-0} =~ ^[01]$ ]] || die "Invalid downgrade option"
  ALLOW_PACKAGE_DOWNGRADES=${ALLOW_PACKAGE_DOWNGRADES:-0}
  TEE_NAME="AMD SEV-SNP"
  TEE_NODE_LABEL_KEY=amd.feature.node.kubernetes.io/snp
  if [[ -z ${NODE_IP:-} ]]; then
    NODE_IP="$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
  fi
  python3 - "$NODE_IP" "$POD_CIDR" "$SERVICE_CIDR" "$CLUSTER_DNS" <<'CHECK'
import ipaddress
import sys
ipaddress.IPv4Address(sys.argv[1])
pod = ipaddress.IPv4Network(sys.argv[2])
service = ipaddress.IPv4Network(sys.argv[3])
assert not pod.overlaps(service), "Pod and service CIDRs overlap"
assert ipaddress.IPv4Address(sys.argv[4]) in service, "Cluster DNS must be in the service subnet"
CHECK
  validate_private_root "$STATE_DIR"
  export NODE_IP TEE_PLATFORM TEE_NAME TEE_NODE_LABEL_KEY RUNTIME_CLASS
  export IGNORE_CHECKSUM_MISMATCH ALLOW_PACKAGE_DOWNGRADES
  mkdir -p "$STATE_DIR"
}


prepare_download_dir() {
  local directory="$STATE_DIR/downloads" uid mode
  [[ $STATE_DIR == /* && $STATE_DIR != / && $STATE_DIR != "$HOME" && $STATE_DIR != /home &&
     $STATE_DIR != /tmp && $STATE_DIR != /var/tmp && $STATE_DIR != *'/../'* && $STATE_DIR != */.. &&
     $STATE_DIR != *'/./'* && $STATE_DIR != */. ]] || die 'Unsafe cache state path'
  # Inspect parents too: a private child can be renamed through a writable
  # parent. Refuse legacy shared caches instead of trusting their contents.
  while [[ $directory != / ]]; do
    [[ ! -L $directory ]] || die "Symlink in cache path: $directory"
    if [[ -e $directory ]]; then
      [[ -d $directory ]] || die "Cache parent is not a directory: $directory"
      uid=$(stat -c %u -- "$directory")
      mode=$(stat -c %a -- "$directory")
      [[ $uid == 0 || $uid == "$EUID" || $uid == "${SUDO_UID:-$EUID}" ]] || die "Untrusted cache owner: $directory"
      if (( (8#$mode & 0022) != 0 )); then
        # Root-owned sticky /tmp-style ancestors cannot have our child renamed
        # by other users. Never allow this exception for the state/cache itself.
        [[ $uid == 0 && $directory != "$STATE_DIR" && $directory != "$STATE_DIR/downloads" ]] &&
          (( (8#$mode & 01000) != 0 )) || die "Writable cache path: $directory; use a fresh private COCO_STATE_DIR"
      fi
    fi
    directory=$(dirname -- "$directory")
  done
  install -d -m 0700 -- "$STATE_DIR" "$STATE_DIR/downloads"
}

validate_download_path() {
  local out=$1 uid mode
  prepare_download_dir
  [[ $(dirname -- "$out") == "$STATE_DIR/downloads" && ! -L $out ]] || die "Unsafe download path: $out"
  if [[ -e $out ]]; then
    [[ -f $out && $(stat -c %h -- "$out") == 1 ]] || die "Download must be a regular, unlinked file: $out"
    uid=$(stat -c %u -- "$out")
    mode=$(stat -c %a -- "$out")
    [[ $uid == 0 || $uid == "$EUID" || $uid == "${SUDO_UID:-$EUID}" ]] || die "Untrusted download owner: $out"
    (( (8#$mode & 0022) == 0 )) || die "Writable cached artifact: $out"
  fi
}



# Populate a persistent cache atomically. A missing/empty artifact is downloaded;
# a verified artifact is also redownloaded whenever its checksum is wrong.
ensure_download() {
  local url=$1 out=$2 partial
  validate_download_path "$out"
  [[ -s "$out" ]] && return 0
  partial=$(mktemp "$STATE_DIR/downloads/.download.XXXXXXXXXX")
  if ! curl -L --fail --silent --show-error "$url" -o "$partial"; then
    rm -f "$partial"
    return 1
  fi
  mv -T -- "$partial" "$out"
}

ensure_download_verified() {
  local url=$1 sha=$2 out=$3 partial
  [[ $sha =~ ^[0-9a-fA-F]{64}$ ]] || die 'Invalid artifact SHA-256'
  validate_download_path "$out"
  if [[ -s "$out" ]]; then
    if echo "$sha  $out" | sha256sum --check --status; then
      return 0
    fi
  fi
  partial=$(mktemp "$STATE_DIR/downloads/.download.XXXXXXXXXX")
  if ! curl -L --fail --silent --show-error "$url" -o "$partial"; then
    rm -f "$partial"
    return 1
  fi
  if ! echo "$sha  $partial" | sha256sum -c -; then
    rm -f "$partial"
    return 1
  fi
  mv -T -- "$partial" "$out"
}

extract_verified_archive() {
  local archive=$1 sha=$2 destination=$3
  [[ $sha =~ ^[0-9a-fA-F]{64}$ ]] || die 'Invalid extraction SHA-256'
  validate_download_path "$archive"
  # Verification and extraction consume the SAME root-private snapshot, not a
  # mutable cache pathname. This also protects a privileged consumer if a cache
  # file is changed after its initial download verification.
  as_root bash -c '
    set -Eeuo pipefail
    umask 077
    snapshot=$(mktemp -d /var/tmp/coco-extract.XXXXXXXXXX)
    trap '\''rm -rf -- "$snapshot"'\'' EXIT
    cp -- "$1" "$snapshot/archive"
    printf "%s  %s\n" "$2" "$snapshot/archive" | sha256sum --check --status
    tar -C "$3" -xzf "$snapshot/archive"
  ' coco-extract "$archive" "$sha" "$destination"
}

install_verified_apt_key() {
  local source=$1 fingerprint=$2 destination=$3
  [[ $fingerprint =~ ^[0-9A-Fa-f]{40}$ ]] || die 'Invalid apt signing-key fingerprint'
  validate_download_path "$source"
  # Convert, inspect and install the SAME private snapshot. Never import into
  # the operator's keyring or accept extra primary keys alongside the pin.
  as_root bash -c '
    set -Eeuo pipefail
    umask 077
    snapshot=$(mktemp -d /var/tmp/coco-apt-key.XXXXXXXXXX)
    trap '\''rm -rf -- "$snapshot"'\'' EXIT
    cp -- "$1" "$snapshot/Release.key"
    install -d -m 0700 "$snapshot/gnupg"
    gpg_args=(--batch --no-options --homedir "$snapshot/gnupg")
    gpg "${gpg_args[@]}" --dearmor <"$snapshot/Release.key" >"$snapshot/key.gpg"
    gpg "${gpg_args[@]}" --with-colons --show-keys --fingerprint "$snapshot/key.gpg" >"$snapshot/keys"
    awk -F: -v expected="${2^^}" '\''
      $1 == "pub" { count++; primary=1 }
      $1 == "sec" || $1 == "ssb" { invalid=1 }
      ($1 == "pub" || $1 == "sub") && ($2 == "r" || $2 == "e") { invalid=1 }
      $1 == "fpr" && primary { fingerprint=$10; primary=0 }
      END { exit !(count == 1 && !invalid && fingerprint == expected) }
    '\'' "$snapshot/keys" || { echo "ERROR: apt signing-key fingerprint mismatch or invalid key bundle" >&2; exit 1; }
    install -m 0644 -- "$snapshot/key.gpg" "$3"
  ' coco-apt-key "$source" "$fingerprint" "$destination"
}

wait_for() {
  local description=$1 timeout=$2; shift 2
  local deadline=$((SECONDS + timeout))
  until "$@"; do
    ((SECONDS < deadline)) || die "Timed out waiting for $description"
    sleep 10
  done
}

require_root_or_sudo() {
  if ((EUID != 0)); then
    need sudo
    sudo -n true ||
      die "Passwordless non-interactive sudo is required; run as root or configure NOPASSWD for this workflow"
  fi
}
