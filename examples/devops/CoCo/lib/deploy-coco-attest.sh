#!/usr/bin/env bash
set -Eeuo pipefail

# Deploy one NVIDIA GPU into an AMD SEV-SNP or Intel TDX Kata CoCo pod,
# collect fresh CPU/GPU attestation evidence, verify it, and save reports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"
POD_NAME="coco-attestation-$(date -u +%Y%m%d%H%M%S | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${PWD}/coco-attestation-reports-$(date -u +%Y%m%dT%H%M%SZ)"
KUBECONFIG_PATH="${KUBECONFIG:-}"
CLEANUP_POD=0
RUNTIME_CLASS="${RUNTIME_CLASS:-kata-qemu-nvidia-gpu-snp}"
TEE_PLATFORM="${TEE_PLATFORM:-snp}"
POD_IMAGE="${POD_IMAGE:-}"
POLICY_REPOSITORY="${POLICY_REPOSITORY:-${POD_IMAGE%@*}}"
REGISTRY_HOST="${REGISTRY_HOST:-${POLICY_REPOSITORY%%/*}}"
GPU_RESOURCE="${GPU_RESOURCE:-nvidia.com/pgpu}"
GPU_COUNT="${GPU_COUNT:-1}"
POD_MEMORY="${POD_MEMORY:-16Gi}"
CONFIDENTIAL_VOLUME_SIZE="${CONFIDENTIAL_VOLUME_SIZE:-8Gi}"
CONFIDENTIAL_VOLUME_MOUNT="/confidential-data"
AMD_KDS_PRODUCT="${AMD_KDS_PRODUCT:-Genoa}"
EXPECTED_SNP_LAUNCH_MEASUREMENT="${EXPECTED_SNP_LAUNCH_MEASUREMENT:-}"
EXPECTED_TDX_MRTD="${EXPECTED_TDX_MRTD:-}"
EXPECTED_TDX_RTMR0="${EXPECTED_TDX_RTMR0:-}"
EXPECTED_TDX_RTMR1="${EXPECTED_TDX_RTMR1:-}"
EXPECTED_TDX_RTMR2="${EXPECTED_TDX_RTMR2:-}"
EXPECTED_TDX_RTMR3="${EXPECTED_TDX_RTMR3:-}"
KBS_NAMESPACE="${KBS_NAMESPACE:-coco-tenant}"
KBS_SERVICE="${KBS_SERVICE:-trustee-kbs}"
IMAGE_POLICY_SOURCE="measured init-data"
COSIGN_PUBLIC_KEY="${COSIGN_PUBLIC_KEY:-}"
REGISTRY_CA_CERT="${REGISTRY_CA_CERT:-}"
ARTIFACT_CACHE_DIR="${ARTIFACT_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:-/var/tmp}/.cache}/bare-metal-coco}"
IGNORE_CHECKSUM_MISMATCH="${IGNORE_CHECKSUM_MISMATCH:-0}"
COSIGN_TLOG_MODE="${COSIGN_TLOG_MODE:-disabled}"

NVAT_VERSION="1.2.2"
NVAT_DEB="nvat-local-repo-ubuntu2404-1-2-local_1.0-1_amd64.deb"
NVAT_URL="https://developer.download.nvidia.com/compute/nvat/1.2.2/local_installers/${NVAT_DEB}"
# SHA-256 of the downloaded NVIDIA .deb package.
NVAT_SHA256="31b0a1646f2bbc08ee599d10dbae106124ef2903f39e37095b96493913b37657"
GO_TDX_GUEST_COMMIT="d0438ad179370160a3b98d9703b1559dcd1ed5ee"
GO_TDX_GUEST_SHA256="5c0a76ad4cc9f780d1dc55cf6f6bd7bccf25d3c8f7b74b05cc478b9001f7b51b"

usage() {
  cat <<'USAGE'
Usage: deploy-coco-attest.sh [options]

Options:
  --namespace NAME       Kubernetes namespace (default: default)
  --pod-name NAME        Pod name (default: timestamped unique name)
  --output-dir DIR       Report directory (default: timestamped directory)
  --kubeconfig FILE      Kubeconfig file (default: $KUBECONFIG/current context)
  --cleanup              Delete the pod after successful attestation
  -h, --help             Show this help

Environment overrides:
  TEE_PLATFORM           snp or tdx
  AMD_KDS_PRODUCT        AMD KDS product name (default: Genoa)
  EXPECTED_SNP_LAUNCH_MEASUREMENT
                         Approved SNP launch measurement (96 hex characters)
  EXPECTED_TDX_MRTD      Approved TDX MRTD (96 hex characters)
  EXPECTED_TDX_RTMR0..3  Approved TDX RTMR values (96 hex each)
  POD_IMAGE              Digest-pinned signed image
  POLICY_REPOSITORY      Repository authorized by the image policy
  REGISTRY_HOST          Registry host:port visible to the guest
  COSIGN_PUBLIC_KEY      Cosign public-key PEM
  REGISTRY_CA_CERT       Private registry CA PEM
  RUNTIME_CLASS          CoCo RuntimeClass
  GPU_RESOURCE           Extended GPU resource (default: nvidia.com/pgpu)
  GPU_COUNT              GPUs assigned to the pod (default: 1)
  POD_MEMORY             Pod memory limit (default: 16Gi)
  CONFIDENTIAL_VOLUME_SIZE
                         Encrypted emptyDir size limit (default: 8Gi)

The pod is privileged only inside its isolated Kata VM so it can access the
guest-local attestation interface. It receives one nvidia.com/pgpu device.
USAGE
}

while (($#)); do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --pod-name) POD_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --kubeconfig) KUBECONFIG_PATH="$2"; shift 2 ;;
    --cleanup) CLEANUP_POD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$POD_IMAGE" == *@sha256:* ]] || { echo "POD_IMAGE must be a digest-pinned signed image" >&2; exit 1; }
[[ -n "$POLICY_REPOSITORY" ]] || { echo "POLICY_REPOSITORY is required" >&2; exit 1; }
[[ -n "$REGISTRY_HOST" ]] || { echo "REGISTRY_HOST is required" >&2; exit 1; }
case "$TEE_PLATFORM" in
  snp)
    TEE_DISPLAY_NAME="AMD SEV-SNP"
    EXPECTED_SNP_LAUNCH_MEASUREMENT="${EXPECTED_SNP_LAUNCH_MEASUREMENT,,}"
    [[ "$EXPECTED_SNP_LAUNCH_MEASUREMENT" =~ ^[0-9a-f]{96}$ ]] || {
      echo "EXPECTED_SNP_LAUNCH_MEASUREMENT must be the approved 48-byte SNP launch measurement (96 hex characters)" >&2
      exit 1
    }
    ;;
  tdx)
    TEE_DISPLAY_NAME="Intel TDX"
    for reference_name in EXPECTED_TDX_MRTD EXPECTED_TDX_RTMR0 EXPECTED_TDX_RTMR1 EXPECTED_TDX_RTMR2 EXPECTED_TDX_RTMR3; do
      printf -v "$reference_name" '%s' "${!reference_name,,}"
      [[ "${!reference_name}" =~ ^[0-9a-f]{96}$ ]] || {
        echo "$reference_name must be an approved 48-byte TDX reference value (96 hex characters)" >&2
        exit 1
      }
    done
    ;;
  *)
    echo "TEE_PLATFORM must be snp or tdx" >&2
    exit 1
    ;;
esac
[[ "$CONFIDENTIAL_VOLUME_SIZE" =~ ^[1-9][0-9]*(Ki|Mi|Gi|Ti)$ ]] || {
  echo "CONFIDENTIAL_VOLUME_SIZE must be a positive binary Kubernetes quantity such as 8Gi" >&2
  exit 1
}

for tool in kubectl curl openssl gcc jq xxd awk sed sha256sum mktemp gzip base64 cosign; do
  command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done
if [[ "$TEE_PLATFORM" == tdx ]]; then
  command -v go >/dev/null || { echo "Missing required command for TDX verification: go" >&2; exit 1; }
fi

K=(kubectl)
if [[ -n "$KUBECONFIG_PATH" ]]; then
  K+=(--kubeconfig "$KUBECONFIG_PATH")
fi

TMP_DIR="$(mktemp -d)"
SUCCESS=0
on_exit() {
  status=$?
  if ((status != 0)); then
    echo "Attestation failed. Pod diagnostics:" >&2
    "${K[@]}" describe pod -n "$NAMESPACE" "$POD_NAME" >&2 2>/dev/null || true
  fi
  if ((CLEANUP_POD == 1 && SUCCESS == 1)); then
    "${K[@]}" delete pod -n "$NAMESPACE" "$POD_NAME" --wait=true >/dev/null || true
  fi
  rm -rf "$TMP_DIR"
  exit "$status"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_DIR"
mkdir -p "$ARTIFACT_CACHE_DIR"
case "$IGNORE_CHECKSUM_MISMATCH" in
  0|1) ;;
  *) echo "IGNORE_CHECKSUM_MISMATCH must be 0 or 1" >&2; exit 1 ;;
esac
case "$COSIGN_TLOG_MODE" in
  disabled|rekor) ;;
  *) echo "COSIGN_TLOG_MODE must be 'disabled' or 'rekor'" >&2; exit 1 ;;
esac
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
MANIFEST="$OUTPUT_DIR/coco-attestation-pod.yaml"

[[ -r "$COSIGN_PUBLIC_KEY" ]] || { echo "Missing Cosign public key: $COSIGN_PUBLIC_KEY" >&2; exit 1; }
[[ -r "$REGISTRY_CA_CERT" ]] || { echo "Missing registry CA certificate: $REGISTRY_CA_CERT" >&2; exit 1; }
COSIGN_PUBLIC_KEY_DATA="$(<"$COSIGN_PUBLIC_KEY")"
REGISTRY_CA_CERT_DATA="$(<"$REGISTRY_CA_CERT")"
IMAGE_SECURITY_POLICY="$(jq -cn \
  --arg repository "$POLICY_REPOSITORY" \
  --arg key "$COSIGN_PUBLIC_KEY_DATA" \
  '{default:[{type:"reject"}],transports:{docker:{($repository):[{type:"sigstoreSigned",keyData:$key,signedIdentity:{type:"matchRepository"}}]}}}')"
echo "Verifying the pinned CUDA image signature from the host ..."
if [[ "$COSIGN_TLOG_MODE" == rekor ]]; then
  if ! cosign verify --allow-insecure-registry \
    --key "$COSIGN_PUBLIC_KEY" "$POD_IMAGE" \
    >/dev/null 2>"$TMP_DIR/cosign-stderr"; then
    cat "$TMP_DIR/cosign-stderr" >&2
    exit 1
  fi
else
  echo "NOTICE: COSIGN_TLOG_MODE=disabled; verifying the private-key signature without public transparency-log inclusion."
  if ! cosign verify --allow-insecure-registry --insecure-ignore-tlog \
    --key "$COSIGN_PUBLIC_KEY" "$POD_IMAGE" \
    >/dev/null 2>"$TMP_DIR/cosign-stderr"; then
    cat "$TMP_DIR/cosign-stderr" >&2
    exit 1
  fi
  sed '/^WARNING: Skipping tlog verification is an insecure practice/d' "$TMP_DIR/cosign-stderr" >&2
fi

KBS_IP="$("${K[@]}" get service -n "$KBS_NAMESPACE" "$KBS_SERVICE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
if [[ -n "$KBS_IP" && "$KBS_IP" != "None" ]]; then
  KBS_ADDRESS="http://${KBS_IP}:8080"
else
  KBS_ADDRESS="not required for public policy/key delivered in measured init-data"
fi

cat >"$OUTPUT_DIR/initdata.toml" <<EOF
version = "0.1.0"
algorithm = "sha256"

[data]
"cdh.toml" = '''
[kbc]
name = "offline_fs_kbc"
url = ""

[image]
image_security_policy = '${IMAGE_SECURITY_POLICY}'
extra_root_certificates = ["""${REGISTRY_CA_CERT_DATA}"""]

[image.registry_config]
unqualified-search-registries = ["docker.io"]

[[image.registry_config.registry]]
location = "${REGISTRY_HOST}"
insecure = false
'''

"policy.rego" = '''
package agent_policy
default AddARPNeighborsRequest := true
default AddSwapRequest := true
default CloseStdinRequest := true
default CopyFileRequest := true
default CreateContainerRequest := true
default CreateSandboxRequest := true
default DestroySandboxRequest := true
default ExecProcessRequest := true
default GetMetricsRequest := true
default GetOOMEventRequest := true
default GuestDetailsRequest := true
default ListInterfacesRequest := true
default ListRoutesRequest := true
default MemHotplugByProbeRequest := true
default OnlineCPUMemRequest := true
default PauseContainerRequest := true
default PullImageRequest := true
default ReadStreamRequest := true
default RemoveContainerRequest := true
default RemoveStaleVirtiofsShareMountsRequest := true
default ReseedRandomDevRequest := true
default ResumeContainerRequest := true
default SetGuestDateTimeRequest := true
default SetPolicyRequest := false
default SignalProcessRequest := true
default StartContainerRequest := true
default StartTracingRequest := true
default StatsContainerRequest := true
default StopTracingRequest := true
default TtyWinResizeRequest := true
default UpdateContainerRequest := true
default UpdateEphemeralMountsRequest := true
default UpdateInterfaceRequest := true
default UpdateRoutesRequest := true
default WaitProcessRequest := true
default WriteStreamRequest := true
'''
EOF
INIT_DATA="$(gzip -c "$OUTPUT_DIR/initdata.toml" | base64 -w 0)"
INITDATA_SHA256="$(sha256sum "$OUTPUT_DIR/initdata.toml" | awk '{print $1}')"

"${K[@]}" get runtimeclass "$RUNTIME_CLASS" >/dev/null
"${K[@]}" get namespace "$NAMESPACE" >/dev/null

cat >"$MANIFEST" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  annotations:
    io.katacontainers.config.hypervisor.cc_init_data: "${INIT_DATA}"
  labels:
    app: coco-attestation-sample
spec:
  runtimeClassName: ${RUNTIME_CLASS}
  restartPolicy: Never
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
    supplementalGroups: [1, 2, 3, 4, 6, 10, 11, 20, 26, 27]
  containers:
    - name: attestation-client
      image: ${POD_IMAGE}
      command: ["/bin/bash", "-lc", "trap : TERM INT; sleep infinity & wait"]
      securityContext:
        privileged: true
        runAsUser: 0
      volumeMounts:
        - name: confidential-scratch
          mountPath: ${CONFIDENTIAL_VOLUME_MOUNT}
      resources:
        limits:
          ${GPU_RESOURCE}: "${GPU_COUNT}"
          memory: ${POD_MEMORY}
  volumes:
    - name: confidential-scratch
      emptyDir:
        sizeLimit: ${CONFIDENTIAL_VOLUME_SIZE}
EOF

cat >"$TMP_DIR/snp_report_collect.c" <<'EOF_C'
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/sev-guest.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#define REPORT_OFFSET 32
#define REPORT_SIZE 1184
static uint32_t le32(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static uint64_t le64(const uint8_t *p){return (uint64_t)le32(p)|((uint64_t)le32(p+4)<<32);}
static void hex(const uint8_t *p,size_t n){for(size_t i=0;i<n;i++)printf("%02x",p[i]);}
static int hv(char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;return -1;}
static int parse64(const char*s,uint8_t out[64]){if(strlen(s)!=128)return-1;for(size_t i=0;i<64;i++){int h=hv(s[i*2]),l=hv(s[i*2+1]);if(h<0||l<0)return-1;out[i]=(h<<4)|l;}return 0;}
static void tcb(const char*n,uint64_t v){printf("%s: boot_loader=%u tee=%u snp=%u microcode=%u (raw=0x%016"PRIx64")\n",n,(unsigned)(v&255),(unsigned)((v>>8)&255),(unsigned)((v>>48)&255),(unsigned)((v>>56)&255),v);}
int main(int ac,char**av){
 if(ac!=4){fprintf(stderr,"usage: %s DEVICE NONCE_HEX RAW_REPORT\n",av[0]);return 2;}
 struct snp_report_req q={0};struct snp_report_resp s={0};
 struct snp_guest_request_ioctl io={.msg_version=1,.req_data=(uint64_t)(uintptr_t)&q,.resp_data=(uint64_t)(uintptr_t)&s};
 if(parse64(av[2],q.user_data)){fprintf(stderr,"nonce must be 128 hex characters\n");return 2;}q.vmpl=0;
 int fd=open(av[1],O_RDWR|O_CLOEXEC);if(fd<0){perror("open sev-guest");return 1;}
 if(ioctl(fd,SNP_GET_REPORT,&io)<0){fprintf(stderr,"SNP_GET_REPORT: %s fw=%u vmm=%u\n",strerror(errno),io.fw_error,io.vmm_error);return 1;}close(fd);
 uint32_t st=le32(s.data),sz=le32(s.data+4);if(st||sz<REPORT_SIZE){fprintf(stderr,"bad response status=%u size=%u\n",st,sz);return 1;}
 const uint8_t*r=s.data+REPORT_OFFSET;FILE*f=fopen(av[3],"wb");if(!f||fwrite(r,1,REPORT_SIZE,f)!=REPORT_SIZE||fclose(f)){perror("write report");return 1;}
 uint64_t p=le64(r+8),pi=le64(r+64);
 puts("AMD SEV-SNP ATTESTATION REPORT\n================================");
 printf("Report version: %u\nGuest SVN: %u\nGuest policy: 0x%016"PRIx64"\n",le32(r),le32(r+4),p);
 printf("  ABI: %u.%u\n  SMT allowed: %s\n  Migration agent allowed: %s\n  Debug allowed: %s\n  Single-socket required: %s\n",(unsigned)((p>>8)&255),(unsigned)(p&255),(p&(1ULL<<16))?"yes":"no",(p&(1ULL<<18))?"yes":"no",(p&(1ULL<<19))?"yes":"no",(p&(1ULL<<20))?"yes":"no");
 printf("Family ID: ");hex(r+16,16);printf("\nImage ID: ");hex(r+32,16);
 printf("\nVMPL: %u\nSignature algorithm: %u (%s)\n",le32(r+48),le32(r+52),le32(r+52)==1?"ECDSA P-384 with SHA-384":"unknown");
 tcb("Current TCB",le64(r+56));printf("Platform info: 0x%016"PRIx64"\n",pi);
 printf("  SMT enabled: %s\n  TSME enabled: %s\n  ECC memory enabled: %s\nAuthor key enabled: %s\n",pi&1?"yes":"no",pi&2?"yes":"no",pi&4?"yes":"no",le32(r+72)?"yes":"no");
 printf("Report data / nonce: ");hex(r+80,64);printf("\nLaunch measurement: ");hex(r+144,48);
 printf("\nHost data: ");hex(r+192,32);printf("\nID key digest: ");hex(r+224,48);
 printf("\nAuthor key digest: ");hex(r+272,48);printf("\nReport ID: ");hex(r+320,32);
 printf("\nMigration-agent report ID: ");hex(r+352,32);printf("\n");tcb("Reported TCB",le64(r+384));
 printf("Chip ID: ");hex(r+416,64);printf("\n");tcb("Committed TCB",le64(r+480));
 printf("Current firmware: API %u.%u build %u\nCommitted firmware: API %u.%u build %u\n",r[490],r[489],r[488],r[494],r[493],r[492]);
 tcb("Launch TCB",le64(r+496));printf("Raw signed report: cpu-attestation-report.bin (%u bytes)\n",REPORT_SIZE);return 0;
}
EOF_C

echo "Deploying $NAMESPACE/$POD_NAME with runtime $RUNTIME_CLASS ..."
"${K[@]}" apply -f "$MANIFEST"
"${K[@]}" wait -n "$NAMESPACE" --for=condition=Ready "pod/$POD_NAME" --timeout=20m

kexec(){ "${K[@]}" exec -n "$NAMESPACE" "$POD_NAME" -- "$@"; }
kcp_to(){ "${K[@]}" cp "$1" "$NAMESPACE/$POD_NAME:$2"; }
kcp_from(){ "${K[@]}" cp "$NAMESPACE/$POD_NAME:$1" "$2"; }
GUEST_TMP="/tmp/coco-attestation-${POD_NAME}"
kexec mkdir -p "$GUEST_TMP"

echo "Verifying the released CoCo LUKS2/dm-crypt emptyDir volume ..."
STORAGE_MOUNT_LINE="$(
  kexec findmnt -n -o SOURCE,FSTYPE,OPTIONS --target "$CONFIDENTIAL_VOLUME_MOUNT"
)"
read -r STORAGE_SOURCE STORAGE_FSTYPE STORAGE_OPTIONS <<<"$STORAGE_MOUNT_LINE"
[[ "$STORAGE_SOURCE" == /dev/mapper/* ]] || {
  echo "Confidential emptyDir is not mounted from a device-mapper path: $STORAGE_SOURCE" >&2
  exit 1
}
[[ "$STORAGE_FSTYPE" == ext4 ]] || {
  echo "Confidential emptyDir has unexpected filesystem type: $STORAGE_FSTYPE" >&2
  exit 1
}
[[ ",$STORAGE_OPTIONS," == *,rw,* ]] || {
  echo "Confidential emptyDir is not writable: $STORAGE_OPTIONS" >&2
  exit 1
}
STORAGE_MAPPER_NAME="${STORAGE_SOURCE##*/}"
STORAGE_DM_NAME="$(
  kexec bash -lc '
    for path in /sys/block/dm-*/dm/name; do
      test -r "$path" || continue
      if test "$(cat "$path")" = "$1"; then
        basename "$(dirname "$(dirname "$path")")"
        exit 0
      fi
    done
    exit 1
  ' _ "$STORAGE_MAPPER_NAME"
)"
STORAGE_DM_DEVICE="/dev/$STORAGE_DM_NAME"
[[ "$STORAGE_DM_NAME" =~ ^dm-[0-9]+$ ]] || {
  echo "Confidential emptyDir mapper did not resolve through guest sysfs: $STORAGE_SOURCE" >&2
  exit 1
}
STORAGE_DM_UUID="$(kexec cat "/sys/block/$STORAGE_DM_NAME/dm/uuid")"
[[ "$STORAGE_DM_UUID" == CRYPT-LUKS2-* ]] || {
  echo "Confidential emptyDir is not a LUKS2/dm-crypt mapping: $STORAGE_DM_UUID" >&2
  exit 1
}
STORAGE_DM_UUIDS="$(
  kexec bash -lc 'for path in /sys/block/dm-*/dm/uuid; do test -r "$path" || continue; printf "%s: " "${path#/sys/block/}"; cat "$path"; done'
)"
grep -q 'CRYPT-LUKS2-' <<<"$STORAGE_DM_UUIDS" || {
  echo "No LUKS2/dm-crypt mapping was found in the guest device-mapper inventory." >&2
  exit 1
}
grep -q 'INTEGRITY-' <<<"$STORAGE_DM_UUIDS" || {
  echo "No dm-integrity mapping was found beneath the confidential volume." >&2
  exit 1
}
STORAGE_CHALLENGE="$(openssl rand -hex 32)"
kexec bash -lc "umask 077; printf '%s\n' '$STORAGE_CHALLENGE' > '$CONFIDENTIAL_VOLUME_MOUNT/encryption-probe'; sync; test \"\$(cat '$CONFIDENTIAL_VOLUME_MOUNT/encryption-probe')\" = '$STORAGE_CHALLENGE'"
STORAGE_CAPACITY="$(kexec df -hP "$CONFIDENTIAL_VOLUME_MOUNT" | tail -n 1)"
cat >"$OUTPUT_DIR/storage-verification-report.txt" <<EOF
COCO CONFIDENTIAL EMPTYDIR VERIFICATION
=======================================
Result: PASS
Pod: ${NAMESPACE}/${POD_NAME}
Runtime class: ${RUNTIME_CLASS}
Configured emptyDir mode: block-encrypted
Volume type: released CoCo confidential emptyDir
Mount point: ${CONFIDENTIAL_VOLUME_MOUNT}
Requested Kubernetes sizeLimit: ${CONFIDENTIAL_VOLUME_SIZE}
Mounted source: ${STORAGE_SOURCE}
Filesystem: ${STORAGE_FSTYPE}
Mount options: ${STORAGE_OPTIONS}
Device-mapper node: ${STORAGE_DM_DEVICE}
LUKS mapper UUID: ${STORAGE_DM_UUID}

Device-mapper inventory:
${STORAGE_DM_UUIDS}

Capacity:
${STORAGE_CAPACITY}

Encryption: LUKS2 / dm-crypt
Integrity: dm-integrity
Key origin: random ephemeral key generated inside the confidential guest
LUKS2 header location: ${TEE_DISPLAY_NAME}-protected guest memory
Persistence: ephemeral; removed with the Pod
Replay protection: not provided
Read/write probe: VERIFIED OK
Overall confidential storage result: PASS
EOF

if [[ "$TEE_PLATFORM" == snp ]]; then
  echo "Compiling and injecting the AMD SNP report collector ..."
  gcc -O2 -Wall -Wextra "$TMP_DIR/snp_report_collect.c" -o "$TMP_DIR/snp_report_collect"
  kcp_to "$TMP_DIR/snp_report_collect" "$GUEST_TMP/snp_report_collect"
  kexec test -s "$GUEST_TMP/snp_report_collect"
  kexec chmod 0755 "$GUEST_TMP/snp_report_collect"

  SEV_DEV="$(kexec cat /sys/class/misc/sev-guest/dev)"
  SEV_MAJOR="${SEV_DEV%%:*}"
  SEV_MINOR="${SEV_DEV##*:}"
  kexec bash -lc "test -e /dev/sev-guest || mknod -m 600 /dev/sev-guest c '$SEV_MAJOR' '$SEV_MINOR'"

  CPU_NONCE="$(openssl rand -hex 64)"
  echo "Collecting AMD SEV-SNP report ..."
  kexec bash -lc "'$GUEST_TMP/snp_report_collect' /dev/sev-guest '$CPU_NONCE' '$GUEST_TMP/cpu-attestation-report.bin' > '$GUEST_TMP/cpu-attestation-report.txt'"
  kexec test -s "$GUEST_TMP/cpu-attestation-report.bin"
  kexec test -s "$GUEST_TMP/cpu-attestation-report.txt"
  kcp_from "$GUEST_TMP/cpu-attestation-report.bin" "$OUTPUT_DIR/cpu-attestation-report.bin"
  kcp_from "$GUEST_TMP/cpu-attestation-report.txt" "$OUTPUT_DIR/cpu-attestation-report.txt"
  grep -q "^Report data / nonce: ${CPU_NONCE}$" "$OUTPUT_DIR/cpu-attestation-report.txt"
  grep -q "^Host data: ${INITDATA_SHA256}$" "$OUTPUT_DIR/cpu-attestation-report.txt"
  OBSERVED_SNP_LAUNCH_MEASUREMENT="$(
    sed -n 's/^Launch measurement: //p' "$OUTPUT_DIR/cpu-attestation-report.txt"
  )"
  [[ "$OBSERVED_SNP_LAUNCH_MEASUREMENT" =~ ^[0-9a-f]{96}$ ]] || {
    echo "SNP report did not contain a valid 48-byte launch measurement" >&2
    exit 1
  }

  CHIP_ID="$(sed -n 's/^Chip ID: //p' "$OUTPUT_DIR/cpu-attestation-report.txt")"
  TCB_LINE="$(sed -n 's/^Reported TCB: //p' "$OUTPUT_DIR/cpu-attestation-report.txt")"
  BL_SPL="$(sed -n 's/.*boot_loader=\([0-9]*\).*/\1/p' <<<"$TCB_LINE")"
  TEE_SPL="$(sed -n 's/.*tee=\([0-9]*\).*/\1/p' <<<"$TCB_LINE")"
  SNP_SPL="$(sed -n 's/.*snp=\([0-9]*\).*/\1/p' <<<"$TCB_LINE")"
  UCODE_SPL="$(sed -n 's/.*microcode=\([0-9]*\).*/\1/p' <<<"$TCB_LINE")"

  echo "Fetching the matching AMD ${AMD_KDS_PRODUCT} VCEK and certificate chain ..."
  curl -L --fail --silent --show-error \
    "https://kdsintf.amd.com/vcek/v1/${AMD_KDS_PRODUCT}/${CHIP_ID}?blSPL=${BL_SPL}&teeSPL=${TEE_SPL}&snpSPL=${SNP_SPL}&ucodeSPL=${UCODE_SPL}" \
    -o "$TMP_DIR/vcek.der"
  curl -L --fail --silent --show-error \
    "https://kdsintf.amd.com/vcek/v1/${AMD_KDS_PRODUCT}/cert_chain" \
    -o "$TMP_DIR/cert-chain.pem"

  R_HEX="$(xxd -p -c 48 -s 672 -l 48 "$OUTPUT_DIR/cpu-attestation-report.bin" | awk '{for(i=length($0)-1;i>=1;i-=2)printf substr($0,i,2)}')"
  S_HEX="$(xxd -p -c 48 -s 744 -l 48 "$OUTPUT_DIR/cpu-attestation-report.bin" | awk '{for(i=length($0)-1;i>=1;i-=2)printf substr($0,i,2)}')"
  cat >"$TMP_DIR/signature.asn1" <<EOF
asn1=SEQUENCE:signature
[signature]
r=INTEGER:0x${R_HEX}
s=INTEGER:0x${S_HEX}
EOF
  openssl asn1parse -genconf "$TMP_DIR/signature.asn1" -out "$TMP_DIR/signature.der" -noout
  head -c 672 "$OUTPUT_DIR/cpu-attestation-report.bin" >"$TMP_DIR/signed-data.bin"
  openssl x509 -inform DER -in "$TMP_DIR/vcek.der" -out "$TMP_DIR/vcek.pem"
  openssl x509 -in "$TMP_DIR/vcek.pem" -pubkey -noout >"$TMP_DIR/vcek-public-key.pem"
  VCEK_CHAIN_RESULT="$(openssl verify -CAfile "$TMP_DIR/cert-chain.pem" "$TMP_DIR/vcek.pem")"
  CPU_SIG_RESULT="$(openssl dgst -sha384 -verify "$TMP_DIR/vcek-public-key.pem" -signature "$TMP_DIR/signature.der" "$TMP_DIR/signed-data.bin")"
  [[ "$VCEK_CHAIN_RESULT" == *": OK" ]]
  [[ "$CPU_SIG_RESULT" == "Verified OK" ]]
  # An authentic report is not sufficient by itself: its launch measurement
  # must also identify an approved VM build.
  if [[ "$OBSERVED_SNP_LAUNCH_MEASUREMENT" != "$EXPECTED_SNP_LAUNCH_MEASUREMENT" ]]; then
    echo "SNP launch measurement does not match the approved reference" >&2
    echo "Expected: $EXPECTED_SNP_LAUNCH_MEASUREMENT" >&2
    echo "Observed: $OBSERVED_SNP_LAUNCH_MEASUREMENT" >&2
    exit 1
  fi
  VCEK_SUBJECT="$(openssl x509 -in "$TMP_DIR/vcek.pem" -noout -subject | sed 's/^subject=//')"
  VCEK_ISSUER="$(openssl x509 -in "$TMP_DIR/vcek.pem" -noout -issuer | sed 's/^issuer=//')"
  VCEK_DATES="$(openssl x509 -in "$TMP_DIR/vcek.pem" -noout -dates | tr '\n' ' ')"
  SIGNED_SHA384="$(openssl dgst -sha384 "$TMP_DIR/signed-data.bin" | sed 's/^.*= //')"
  CPU_RAW_SHA256="$(sha256sum "$OUTPUT_DIR/cpu-attestation-report.bin" | awk '{print $1}')"
  cat >>"$OUTPUT_DIR/cpu-attestation-report.txt" <<EOF

CRYPTOGRAPHIC VERIFICATION
==========================
Nonce freshness: MATCHES generated 64-byte challenge
Expected SNP launch measurement: ${EXPECTED_SNP_LAUNCH_MEASUREMENT}
Observed SNP launch measurement: ${OBSERVED_SNP_LAUNCH_MEASUREMENT}
SNP launch measurement reference: VERIFIED OK
Measured init-data SHA-256: ${INITDATA_SHA256}
SNP HOST_DATA binding: VERIFIED OK
VCEK subject: ${VCEK_SUBJECT}
VCEK issuer: ${VCEK_ISSUER}
VCEK validity: ${VCEK_DATES}
AMD VCEK certificate chain: VERIFIED OK
Report signature (ECDSA P-384 / SHA-384): VERIFIED OK
Signed data SHA-384: ${SIGNED_SHA384}
Raw report SHA-256: ${CPU_RAW_SHA256}
Overall CPU attestation result: PASS
EOF
else
  echo "Building the pinned Intel TDX quote collector and verifier ..."
  TDX_ARCHIVE="$ARTIFACT_CACHE_DIR/go-tdx-guest-${GO_TDX_GUEST_COMMIT}.tar.gz"
  TDX_DOWNLOAD=0
  if [[ ! -s "$TDX_ARCHIVE" ]]; then
    TDX_DOWNLOAD=1
  elif ! echo "${GO_TDX_GUEST_SHA256}  $TDX_ARCHIVE" | sha256sum --check --status; then
    if [[ "$IGNORE_CHECKSUM_MISMATCH" == 1 ]]; then
      echo "WARNING: ignoring checksum mismatch for cached TDX verifier source $TDX_ARCHIVE" >&2
    else
      TDX_DOWNLOAD=1
    fi
  fi
  if ((TDX_DOWNLOAD == 1)); then
    TDX_PARTIAL="${TDX_ARCHIVE}.partial.$$"
    curl -L --fail --silent --show-error \
      "https://github.com/google/go-tdx-guest/archive/${GO_TDX_GUEST_COMMIT}.tar.gz" \
      -o "$TDX_PARTIAL"
    if ! echo "${GO_TDX_GUEST_SHA256}  $TDX_PARTIAL" | sha256sum -c -; then
      if [[ "$IGNORE_CHECKSUM_MISMATCH" != 1 ]]; then
        rm -f "$TDX_PARTIAL"
        exit 1
      fi
      echo "WARNING: ignoring checksum mismatch for downloaded TDX verifier source" >&2
    fi
    mv "$TDX_PARTIAL" "$TDX_ARCHIVE"
  fi
  tar -C "$TMP_DIR" -xzf "$TDX_ARCHIVE"
  TDX_SOURCE_DIR="$TMP_DIR/go-tdx-guest-${GO_TDX_GUEST_COMMIT}"
  (
    cd "$TDX_SOURCE_DIR"
    CGO_ENABLED=0 go build -trimpath -o "$TMP_DIR/tdx-attest" ./tools/attest
    CGO_ENABLED=0 go build -trimpath -o "$TMP_DIR/tdx-check" ./tools/check
  )
  kcp_to "$TMP_DIR/tdx-attest" "$GUEST_TMP/tdx-attest"
  kexec chmod 0755 "$GUEST_TMP/tdx-attest"
  if ! kexec mountpoint -q /sys/kernel/config >/dev/null 2>&1; then
    kexec mount -t configfs configfs /sys/kernel/config
  fi
  kexec test -r /sys/kernel/config/tsm/report

  CPU_NONCE="$(openssl rand -hex 64)"
  echo "Collecting a fresh Intel TDX quote ..."
  kexec "$GUEST_TMP/tdx-attest" \
    -in "$CPU_NONCE" -inform hex -outform bin \
    -out "$GUEST_TMP/cpu-attestation-report.bin"
  if ! kexec test -s "$GUEST_TMP/cpu-attestation-report.bin"; then
    echo "Intel QGS returned an empty quote; verify PCCS has a valid Intel PCS subscription key and platform PCK collateral" >&2
    exit 1
  fi
  kcp_from "$GUEST_TMP/cpu-attestation-report.bin" "$OUTPUT_DIR/cpu-attestation-report.bin"

  MRCONFIG_EXPECTED="${INITDATA_SHA256}00000000000000000000000000000000"
  RTMR_REFERENCES="${EXPECTED_TDX_RTMR0},${EXPECTED_TDX_RTMR1},${EXPECTED_TDX_RTMR2},${EXPECTED_TDX_RTMR3}"
  echo "Verifying TDX quote signature, Intel collateral, freshness, and reference values ..."
  TDX_CHECK_RESULT="$(
    "$TMP_DIR/tdx-check" \
      -in "$OUTPUT_DIR/cpu-attestation-report.bin" -inform bin \
      -get_collateral=true -check_crl=true \
      -mr_td "$EXPECTED_TDX_MRTD" \
      -mr_config_id "$MRCONFIG_EXPECTED" \
      -rtmrs "$RTMR_REFERENCES" \
      -report_data "$CPU_NONCE"
  )"
  # With set -e, the assignment above fails immediately when the verifier
  # returns a nonzero verification, network, or policy exit code. Do not infer
  # success from its human-readable message, whose capitalization can change.

  quote_hex() {
    xxd -p -c 256 -s "$1" -l "$2" "$OUTPUT_DIR/cpu-attestation-report.bin" |
      tr -d '\n'
  }
  QUOTE_VERSION_HEX="$(quote_hex 0 2)"
  case "$QUOTE_VERSION_HEX" in
    0400) QUOTE_BODY_SHIFT=0 ;;
    # Quote v5 inserts a 2-byte body type and 4-byte body-size field between
    # the common 48-byte header and the otherwise shared TDX 1.0 body.
    0500) QUOTE_BODY_SHIFT=6 ;;
    *)
      echo "Unsupported Intel TD Quote version bytes: $QUOTE_VERSION_HEX" >&2
      exit 1
      ;;
  esac
  OBSERVED_TDX_MRTD="$(quote_hex $((184 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_MRCONFIGID="$(quote_hex $((232 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_RTMR0="$(quote_hex $((376 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_RTMR1="$(quote_hex $((424 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_RTMR2="$(quote_hex $((472 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_RTMR3="$(quote_hex $((520 + QUOTE_BODY_SHIFT)) 48)"
  OBSERVED_TDX_REPORTDATA="$(quote_hex $((568 + QUOTE_BODY_SHIFT)) 64)"
  TDX_ATTRIBUTES="$(quote_hex $((168 + QUOTE_BODY_SHIFT)) 8)"
  (( (16#${TDX_ATTRIBUTES:0:2} & 1) == 0 )) || {
    echo "TDX quote reports a debug-enabled Trust Domain" >&2
    exit 1
  }
  [[ "$OBSERVED_TDX_MRTD" == "$EXPECTED_TDX_MRTD" ]]
  [[ "$OBSERVED_TDX_MRCONFIGID" == "$MRCONFIG_EXPECTED" ]]
  [[ "$OBSERVED_TDX_RTMR0" == "$EXPECTED_TDX_RTMR0" ]]
  [[ "$OBSERVED_TDX_RTMR1" == "$EXPECTED_TDX_RTMR1" ]]
  [[ "$OBSERVED_TDX_RTMR2" == "$EXPECTED_TDX_RTMR2" ]]
  [[ "$OBSERVED_TDX_RTMR3" == "$EXPECTED_TDX_RTMR3" ]]
  [[ "$OBSERVED_TDX_REPORTDATA" == "$CPU_NONCE" ]]
  CPU_RAW_SHA256="$(sha256sum "$OUTPUT_DIR/cpu-attestation-report.bin" | awk '{print $1}')"
  cat >"$OUTPUT_DIR/cpu-attestation-report.txt" <<EOF
INTEL TDX ATTESTATION REPORT
============================
Pod: ${NAMESPACE}/${POD_NAME}
Runtime class: ${RUNTIME_CLASS}
Quote version (little-endian): ${QUOTE_VERSION_HEX}
TD attributes: ${TDX_ATTRIBUTES}
Debug enabled: no
MRTD: ${OBSERVED_TDX_MRTD}
MRCONFIGID: ${OBSERVED_TDX_MRCONFIGID}
RTMR0: ${OBSERVED_TDX_RTMR0}
RTMR1: ${OBSERVED_TDX_RTMR1}
RTMR2: ${OBSERVED_TDX_RTMR2}
RTMR3: ${OBSERVED_TDX_RTMR3}
Report data / nonce: ${OBSERVED_TDX_REPORTDATA}

CRYPTOGRAPHIC AND REFERENCE VERIFICATION
========================================
Nonce freshness: MATCHES generated 64-byte challenge
Measured init-data SHA-256: ${INITDATA_SHA256}
TDX MRCONFIGID binding: VERIFIED OK
TDX MRTD reference: VERIFIED OK
TDX RTMR0 reference: VERIFIED OK
TDX RTMR1 reference: VERIFIED OK
TDX RTMR2 reference: VERIFIED OK
TDX RTMR3 reference: VERIFIED OK
TDX debug attribute: DISABLED
Quote signature and embedded certificate chain: VERIFIED OK
Intel PCS collateral, TCB status, and CRLs: VERIFIED OK
Verifier: google/go-tdx-guest ${GO_TDX_GUEST_COMMIT}
Verifier output: ${TDX_CHECK_RESULT}
Raw quote SHA-256: ${CPU_RAW_SHA256}
Overall CPU attestation result: PASS
EOF
fi

echo "Downloading and verifying NVIDIA NVAT ${NVAT_VERSION} ..."
NVAT_PATH="$ARTIFACT_CACHE_DIR/$NVAT_DEB"
NVAT_DOWNLOAD=0
if [[ ! -s "$NVAT_PATH" ]]; then
  NVAT_DOWNLOAD=1
elif ! echo "${NVAT_SHA256}  $NVAT_PATH" | sha256sum --check --status; then
  if [[ "$IGNORE_CHECKSUM_MISMATCH" == 1 ]]; then
    echo "WARNING: ignoring checksum mismatch for cached NVAT package $NVAT_PATH" >&2
  else
    NVAT_DOWNLOAD=1
  fi
fi
if ((NVAT_DOWNLOAD == 1)); then
  NVAT_PARTIAL="${NVAT_PATH}.partial.$$"
  if ! curl -L --fail --silent --show-error "$NVAT_URL" -o "$NVAT_PARTIAL"; then
    rm -f "$NVAT_PARTIAL"
    exit 1
  fi
  if ! echo "${NVAT_SHA256}  $NVAT_PARTIAL" | sha256sum -c -; then
    if [[ "$IGNORE_CHECKSUM_MISMATCH" == 1 ]]; then
      echo "WARNING: ignoring checksum mismatch for downloaded NVAT package $NVAT_URL" >&2
    else
      rm -f "$NVAT_PARTIAL"
      exit 1
    fi
  fi
  mv "$NVAT_PARTIAL" "$NVAT_PATH"
fi
kcp_to "$NVAT_PATH" "$GUEST_TMP/nvat-local-repo.deb"
kexec test -s "$GUEST_TMP/nvat-local-repo.deb"
kexec bash -lc "dpkg -i '$GUEST_TMP/nvat-local-repo.deb' >/dev/null; keyring=\$(find /var/nvat-local-repo-* -maxdepth 1 -name '*-keyring.gpg' -print -quit); test -s \"\$keyring\"; cp \"\$keyring\" /usr/share/keyrings/; apt-get update >/dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y nvattest >/dev/null"

GPU_NONCE="$(openssl rand -hex 32)"
echo "Collecting and locally verifying NVIDIA GPU evidence ..."
kexec bash -lc "nvattest --format json --log-level warn collect-evidence --device gpu --nonce '$GPU_NONCE' > '$GUEST_TMP/gpu-attestation-evidence.json'"
kexec bash -lc "nvattest --format json --log-level warn attest --device gpu --verifier local --nonce '$GPU_NONCE' > '$GUEST_TMP/gpu-attestation-result.json'"
kexec test -s "$GUEST_TMP/gpu-attestation-evidence.json"
kexec test -s "$GUEST_TMP/gpu-attestation-result.json"
kcp_from "$GUEST_TMP/gpu-attestation-evidence.json" "$OUTPUT_DIR/gpu-attestation-evidence.json"
kcp_from "$GUEST_TMP/gpu-attestation-result.json" "$OUTPUT_DIR/gpu-attestation-result.json"
kexec nvidia-smi conf-compute -q >"$OUTPUT_DIR/gpu-confidential-compute-status.txt"
GPU_IDENTITY="$(kexec nvidia-smi --query-gpu=uuid,name,driver_version,vbios_version --format=csv,noheader)"

jq -e --arg nonce "$GPU_NONCE" '
  .result_code == 0 and .result_message == "Ok" and
  .claims[0].eat_nonce == $nonce and
  .claims[0].measres == "success" and
  .claims[0].secboot == true and
  .claims[0].dbgstat == "disabled" and
  .claims[0]."x-nvidia-gpu-attestation-report-nonce-match" == true and
  .claims[0]."x-nvidia-gpu-attestation-report-signature-verified" == true and
  .claims[0]."x-nvidia-gpu-attestation-report-cert-chain"."x-nvidia-cert-status" == "valid" and
  .claims[0]."x-nvidia-gpu-attestation-report-cert-chain"."x-nvidia-cert-ocsp-status" == "good" and
  .claims[0]."x-nvidia-gpu-driver-rim-signature-verified" == true and
  .claims[0]."x-nvidia-gpu-driver-rim-version-match" == true and
  .claims[0]."x-nvidia-gpu-vbios-rim-signature-verified" == true and
  .claims[0]."x-nvidia-gpu-vbios-rim-version-match" == true and
  .claims[0]."x-nvidia-mismatch-measurement-records" == null
' "$OUTPUT_DIR/gpu-attestation-result.json" >/dev/null

GPU_EVIDENCE_SHA256="$(sha256sum "$OUTPUT_DIR/gpu-attestation-evidence.json" | awk '{print $1}')"
GPU_RESULT_SHA256="$(sha256sum "$OUTPUT_DIR/gpu-attestation-result.json" | awk '{print $1}')"
COLLECTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  cat <<EOF
NVIDIA GPU ATTESTATION REPORT
=============================
Collected at: ${COLLECTED_AT}
Collector/verifier: NVIDIA NVAT ${NVAT_VERSION}, local verifier
Pod: ${NAMESPACE}/${POD_NAME}
Runtime class: ${RUNTIME_CLASS}
GPU identity: ${GPU_IDENTITY}

EOF
  jq -r --arg nonce "$GPU_NONCE" '
    .claims[0] as $c |
    $c."x-nvidia-gpu-driver-version" as $driver_version |
    $c."x-nvidia-gpu-vbios-version" as $vbios_version |
    $c."x-nvidia-gpu-attestation-report-nonce-match" as $nonce_match |
    $c."x-nvidia-gpu-attestation-report-parsed" as $report_parsed |
    $c."x-nvidia-gpu-attestation-report-signature-verified" as $report_signature |
    $c."x-nvidia-gpu-attestation-report-cert-chain" as $cert_chain |
    $cert_chain."x-nvidia-cert-status" as $cert_status |
    $cert_chain."x-nvidia-cert-ocsp-status" as $ocsp_status |
    $c."x-nvidia-gpu-attestation-report-cert-chain-fwid-match" as $fwid_match |
    $c."x-nvidia-gpu-arch-check" as $arch_check |
    $c."x-nvidia-gpu-driver-rim-signature-verified" as $driver_rim_signature |
    $c."x-nvidia-gpu-driver-rim-version-match" as $driver_rim_version |
    $c."x-nvidia-gpu-vbios-rim-signature-verified" as $vbios_rim_signature |
    $c."x-nvidia-gpu-vbios-rim-version-match" as $vbios_rim_version |
    $c."x-nvidia-mismatch-measurement-records" as $mismatches |
    "Hardware model claim: \($c.hwmodel)\n" +
    "Driver version: \($driver_version)\n" +
    "VBIOS version: \($vbios_version)\n" +
    "Secure boot claim: \($c.secboot)\n" +
    "Debug state claim: \($c.dbgstat)\n\n" +
    "FRESHNESS\n---------\n" +
    "Challenge nonce: \($nonce)\n" +
    "Evidence nonce match: \($nonce_match)\n\n" +
    "CRYPTOGRAPHIC AND MEASUREMENT VERIFICATION\n------------------------------------------\n" +
    "Attestation report parsed: \($report_parsed)\n" +
    "Attestation report signature verified: \($report_signature)\n" +
    "Attestation certificate status / OCSP: \($cert_status) / \($ocsp_status)\n" +
    "Firmware identity matched certificate: \($fwid_match)\n" +
    "GPU architecture check: \($arch_check)\n" +
    "Measurement appraisal: \($c.measres)\n" +
    "Driver RIM signature/version verified: \($driver_rim_signature) / \($driver_rim_version)\n" +
    "VBIOS RIM signature/version verified: \($vbios_rim_signature) / \($vbios_rim_version)\n" +
    "Mismatched measurement records: \($mismatches)\n\n" +
    "NVAT result: \(.result_code) / \(.result_message)\n" +
    "Overall GPU attestation result: PASS\n"
  ' "$OUTPUT_DIR/gpu-attestation-result.json"
  cat <<EOF

EVIDENCE ARTIFACTS
------------------
Raw signed evidence: gpu-attestation-evidence.json
Verified claims and detached EAT: gpu-attestation-result.json
Raw evidence SHA-256: ${GPU_EVIDENCE_SHA256}
Verified result SHA-256: ${GPU_RESULT_SHA256}
EOF
} >"$OUTPUT_DIR/gpu-attestation-report.txt"

if [[ "$TEE_PLATFORM" == snp ]]; then
  IMAGE_TEE_BINDING="SNP HOST_DATA binding: VERIFIED OK
SNP launch measurement reference: VERIFIED OK"
  IMAGE_ENFORCEMENT_POINT="image-rs inside the Kata SEV-SNP guest"
  IMAGE_READY_EXPLANATION="from SNP-bound init-data"
else
  IMAGE_TEE_BINDING="TDX MRCONFIGID binding: VERIFIED OK
TDX MRTD and RTMR references: VERIFIED OK"
  IMAGE_ENFORCEMENT_POINT="image-rs inside the Kata TDX guest"
  IMAGE_READY_EXPLANATION="from TDX MRCONFIGID-bound init-data"
fi
cat >"$OUTPUT_DIR/image-verification-report.txt" <<EOF
CONTAINER IMAGE SIGNATURE VERIFICATION
======================================
Result: PASS
Image: ${POD_IMAGE}
Policy source: ${IMAGE_POLICY_SOURCE}
Policy default: reject
Signature rule: sigstoreSigned
Public key: embedded as keyData in measured init-data
Registry configuration: embedded in measured init-data
Registry transport: HTTPS with an embedded private root CA
Measured init-data SHA-256: ${INITDATA_SHA256}
${IMAGE_TEE_BINDING}
Available Trustee endpoint: ${KBS_ADDRESS}
Transparency-log mode: ${COSIGN_TLOG_MODE}
Enforcement point: ${IMAGE_ENFORCEMENT_POINT}

The pod reached Ready only after image-rs loaded the deny-by-default policy
${IMAGE_READY_EXPLANATION} and verified the image's Cosign signature.
EOF

(
  cd "$OUTPUT_DIR"
  sha256sum \
    cpu-attestation-report.txt \
    cpu-attestation-report.bin \
    gpu-attestation-report.txt \
    gpu-attestation-evidence.json \
    gpu-attestation-result.json \
    image-verification-report.txt \
    storage-verification-report.txt \
    >SHA256SUMS
)

SUCCESS=1
echo
echo "CPU attestation: PASS"
echo "GPU attestation: PASS"
echo "Confidential storage: PASS"
echo "Reports saved in: $OUTPUT_DIR"
echo "Human-readable reports:"
echo "  $OUTPUT_DIR/cpu-attestation-report.txt"
echo "  $OUTPUT_DIR/gpu-attestation-report.txt"
echo "  $OUTPUT_DIR/image-verification-report.txt"
echo "  $OUTPUT_DIR/storage-verification-report.txt"
echo "Pod: $NAMESPACE/$POD_NAME ($([[ $CLEANUP_POD == 1 ]] && echo deleted || echo left running))"
