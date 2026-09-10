#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config

failures=0
warnings=0
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings+1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures+1)); }
has() { command -v "$1" >/dev/null 2>&1; }

echo "CoCo ${TEE_NAME} + NVIDIA GPU host prerequisite report"
echo "Node address: $NODE_IP"
echo "Selected runtime: $RUNTIME_CLASS"

root_write_probe="$(as_root mktemp -p /etc .bare-metal-coco-write-check.XXXXXX 2>/dev/null || true)"
if [[ -n "$root_write_probe" ]]; then
  as_root rm -f -- "$root_write_probe"
  pass "Root filesystem accepts writes"
else
  fail "Root filesystem is not writable; repair the filesystem/device before installation"
fi

if has findmnt && has lsblk; then
  root_source="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
  root_kname="$(lsblk -n -o KNAME "$root_source" 2>/dev/null | tail -n 1 || true)"
  ext4_errors_file="/sys/fs/ext4/${root_kname}/errors_count"
  if [[ -r "$ext4_errors_file" ]]; then
    ext4_errors="$(<"$ext4_errors_file")"
    if [[ "$ext4_errors" =~ ^[0-9]+$ ]] && ((ext4_errors > 0)); then
      fail "Root ext4 filesystem reports ${ext4_errors} error(s); inspect storage and run an offline fsck"
    else
      pass "Root ext4 filesystem reports no errors"
    fi
  fi
fi

[[ "$(uname -m)" == x86_64 ]] && pass "x86_64 architecture" || fail "x86_64 is required"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu ]] && pass "Ubuntu ${VERSION_ID:-unknown}" || warn "Scripts target Ubuntu; found ${PRETTY_NAME:-unknown}"
else
  fail "Cannot identify operating system"
fi

case "$TEE_PLATFORM" in
  snp)
    grep -q AuthenticAMD /proc/cpuinfo && pass "AMD processor" || fail "AMD processor not detected"
    grep -qw sev_snp /proc/cpuinfo &&
      pass "SEV-SNP CPU flag" ||
      fail "SEV-SNP flag missing; enable SVM/SEV/SNP in BIOS and use a supported CPU/kernel"
    [[ -e /sys/module/kvm_amd/parameters/sev ]] &&
      grep -qiE '^(1|Y)$' /sys/module/kvm_amd/parameters/sev &&
      pass "kvm_amd SEV enabled" ||
      fail "kvm_amd SEV is not enabled"
    [[ -e /dev/sev ]] && pass "/dev/sev present" || fail "/dev/sev missing"
    ;;
  tdx)
    grep -q GenuineIntel /proc/cpuinfo && pass "Intel processor" || fail "Intel processor not detected"
    grep -qw tdx_host_platform /proc/cpuinfo &&
      pass "TDX host-platform CPU flag" ||
      fail "tdx_host_platform flag missing; enable TME/TME-MT/TDX in firmware and use a supported kernel"
    [[ -e /sys/module/kvm_intel/parameters/tdx ]] &&
      grep -qiE '^(1|Y)$' /sys/module/kvm_intel/parameters/tdx &&
      pass "kvm_intel TDX enabled" ||
      fail "kvm_intel TDX is not enabled"
    # Some supported kernels do not retain or emit Canonical's optional
    # "virt/tdx: module initialized" log line. The exported KVM parameter
    # above is the stable capability interface; do not gate on dmesg wording.
    [[ -e /dev/kvm ]] && pass "/dev/kvm present" || fail "/dev/kvm missing"
    ;;
esac
[[ -d /sys/kernel/iommu_groups ]] && [[ -n "$(find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] \
  && pass "IOMMU groups populated" || fail "IOMMU groups absent; enable the platform IOMMU in firmware/kernel"

if has lspci && lspci -Dnn -d 10de: | grep -qE 'VGA compatible controller|3D controller'; then
  gpu_line="$(lspci -Dnn -d 10de: | grep -m1 -E 'VGA compatible controller|3D controller')"
  pass "NVIDIA PCI device: $gpu_line"
  if [[ "$gpu_line" =~ GH100|H100|H200|B200|GB200|Blackwell ]]; then
    pass "GPU family advertises confidential-computing support"
  else
    warn "GPU is not recognized as H100/H200/B200/GB200; confirm confidential-computing support"
  fi
  gpu_bdf="${gpu_line%% *}"
  driver_link="/sys/bus/pci/devices/0000:${gpu_bdf#0000:}/driver"
  driver=""
  [[ ! -L "$driver_link" ]] ||
    driver="$(basename "$(readlink -f "$driver_link")")"
  case "$driver" in
    vfio-pci) pass "GPU is already bound to vfio-pci (binding is validated after GPU Operator installation)" ;;
    "") pass "GPU is currently unbound; this is valid before GPU Operator installation" ;;
    nvidia|nouveau) fail "GPU is bound to conflicting host driver $driver; remove it before installation" ;;
    *) warn "GPU is bound to $driver; stage 20 will require GPU Operator VFIO Manager to replace it with vfio-pci" ;;
  esac
  iommu_group="$(readlink -f "/sys/bus/pci/devices/$gpu_bdf/iommu_group" 2>/dev/null || true)"
  if [[ -n "$iommu_group" ]]; then
    group_members="$(find "$iommu_group/devices" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | wc -l)"
    ((group_members == 1)) && pass "GPU has an isolated IOMMU group" || warn "GPU IOMMU group has $group_members functions; confirm all can be assigned together"
  else
    fail "GPU has no IOMMU group"
  fi
else
  fail "NVIDIA GPU not detected"
fi

if lsmod | awk '$1 ~ /^(nvidia(_|$)|nouveau$)/{found=1} END{exit !found}'; then
  fail "A host NVIDIA/nouveau kernel module is loaded; GPU passthrough requires VFIO instead"
else
  pass "No host NVIDIA or nouveau kernel module loaded"
fi
if has dpkg-query && dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' 'nvidia-driver-*' 2>/dev/null | grep -q '^.i '; then
  fail "Host NVIDIA driver packages are installed; remove them before GPU passthrough"
else
  pass "No host NVIDIA driver package installed"
fi

if find /sys/kernel/iommu_groups -type l 2>/dev/null | grep -q .; then pass "PCI devices assigned to IOMMU groups"; fi
grep -qw swap /proc/swaps 2>/dev/null && warn "Swap is enabled; Kubernetes installer will disable it" || pass "Swap disabled"
mem_gib=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
((mem_gib >= 64)) && pass "Memory: ${mem_gib} GiB" || warn "Only ${mem_gib} GiB RAM; 64 GiB or more is recommended"
disk_gib="$(df -Pk "$SUITE_DIR" | awk 'NR==2{print int($4/1024/1024)}')"
((disk_gib >= 50)) && pass "Free disk: ${disk_gib} GiB" || warn "Only ${disk_gib} GiB free; at least 50 GiB is recommended"

endpoints=(
  https://registry.k8s.io/v2/ \
  https://pkgs.k8s.io/ \
  https://github.com/ \
  https://raw.githubusercontent.com/ \
  https://ghcr.io/v2/ \
  https://helm.ngc.nvidia.com/ \
  https://nvcr.io/v2/ \
  https://quay.io/v2/ \
  https://registry-1.docker.io/v2/ \
  https://developer.download.nvidia.com/
)
if [[ "$TEE_PLATFORM" == snp ]]; then
  endpoints+=(https://kdsintf.amd.com/)
else
  endpoints+=(
    https://api.trustedservices.intel.com/
    https://ppa.launchpadcontent.net/
    https://keyserver.ubuntu.com/
  )
fi
for endpoint in "${endpoints[@]}"; do
  if curl -L -I --connect-timeout 8 --max-time 15 --silent "$endpoint" >/dev/null; then pass "Reachable: $endpoint"; else fail "Cannot reach: $endpoint"; fi
done

echo
echo "Summary: $failures failure(s), $warnings warning(s)"
((failures == 0)) || exit 1
