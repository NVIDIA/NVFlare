#!/usr/bin/env bash
# Read-only checks. Firmware, host kernel, SGX registration and the Intel DCAP
# repository are platform-administrator responsibilities, not guessed defaults.

_runtime_root() {
  if declare -F as_root >/dev/null; then as_root "$@";
  elif (( EUID == 0 )); then "$@";
  else sudo -n "$@"; fi
}

_runtime_fail() { printf 'Runtime prerequisite failed: %s\n' "$*" >&2; return 1; }

validate_runtime_prerequisites() {
  local runtime=$1 config=${2:-} observed port listeners expected
  case "$runtime" in
    kata-qemu-snp|kata-qemu-nvidia-gpu-snp)
      [[ $(_runtime_root cat /sys/module/kvm_amd/parameters/sev_snp) == Y ]] || {
        _runtime_fail 'Enable AMD SEV-SNP firmware and a matching kvm_amd host kernel first'; return 1;
      }
      ;;
    kata-qemu-tdx|kata-qemu-nvidia-gpu-tdx)
      [[ $(_runtime_root cat /sys/module/kvm_intel/parameters/tdx) =~ ^(Y|1)$ ]] || {
        _runtime_fail 'Enable Intel TDX firmware and a matching kvm_intel host kernel first'; return 1;
      }
      [[ -c /dev/kvm ]] || { _runtime_fail 'Missing host /dev/kvm'; return 1; }
      [[ -n ${QGS_PACKAGE_VERSION:-} ]] || {
        _runtime_fail 'Set QGS_PACKAGE_VERSION to the reviewed exact tdx-qgs Debian package version'; return 1;
      }
      observed=$(dpkg-query -W -f='${Version}' tdx-qgs 2>/dev/null) || {
        _runtime_fail "Install Intel DCAP tdx-qgs=${QGS_PACKAGE_VERSION} from the reviewed Intel repository first"; return 1;
      }
      [[ $observed == "$QGS_PACKAGE_VERSION" ]] || {
        _runtime_fail "tdx-qgs version $observed differs from pinned $QGS_PACKAGE_VERSION"; return 1;
      }
      for expected in QGS_CONFIG_SHA256 QGS_QCNL_CONFIG_SHA256; do
        [[ ${!expected:-} =~ ^[0-9a-f]{64}$ ]] || {
          _runtime_fail "Set $expected after reviewing QGS transport and Intel collateral configuration"; return 1;
        }
      done
      printf '%s  %s\n' "$QGS_CONFIG_SHA256" /etc/qgs.conf \
        "$QGS_QCNL_CONFIG_SHA256" /etc/sgx_default_qcnl.conf | _runtime_root sha256sum --check --strict || return 1
      _runtime_root systemctl is-active --quiet qgsd || {
        _runtime_fail 'Configure /etc/qgs.conf and /etc/sgx_default_qcnl.conf, register the platform for PCK collateral, and start qgsd'; return 1;
      }
      # Stage 00 can run before Kata is installed. Stage 04/20 additionally
      # checks the pinned runtime endpoint, so an active but unusable QGS fails.
      if [[ -n $config ]]; then
        port=$(_runtime_root python3 - "$config" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as stream:
    port = tomllib.load(stream)["hypervisor"]["qemu"].get("tdx_quote_generation_service_socket_port", 4050)
if type(port) is not int or not 0 <= port <= 65535:
    raise SystemExit("Invalid TDX quote-generation service port")
print(port)
PY
        ) || return 1
        if [[ $port == 0 ]]; then
          _runtime_root test -S /var/run/tdx-qgs/qgs.socket || {
            _runtime_fail 'Pinned runtime needs QGS Unix socket /var/run/tdx-qgs/qgs.socket'; return 1;
          }
        else
          listeners=$(_runtime_root ss --vsock --listening --numeric) || return 1
          grep -Eq "(^|[[:space:]])(2|\\*):${port}([[:space:]]|$)" <<<"$listeners" || {
            _runtime_fail "Configure and restart qgsd to listen on host VSOCK CID 2 port $port (pinned Kata endpoint)"; return 1;
          }
        fi
      fi
      ;;
    *) _runtime_fail "Unsupported RuntimeClass: $runtime"; return 1 ;;
  esac
}
