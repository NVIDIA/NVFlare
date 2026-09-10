#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 2 ]] || {
    printf 'Usage: %s BASE-CONFIG APPROVAL-ENV\n' "$0" >&2
    exit 2
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="$(realpath -- "$1")"
APPROVAL_ENV="$(realpath -- "$2")"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
for command in awk base64 cmp curl cut docker grep gzip install ip kubectl openssl \
    python3 realpath sed sha256sum skopeo sleep sort tar; do need "${command}"; done
command -v ctr >/dev/null 2>&1 || sudo test -x /usr/bin/ctr || die 'missing command: ctr'
command -v crictl >/dev/null 2>&1 || die 'missing command: crictl (install cri-tools)'
[[ -s "${BASE_CONFIG}" ]] || die "missing base configuration: ${BASE_CONFIG}"
[[ -s "${APPROVAL_ENV}" ]] || die "missing approval configuration: ${APPROVAL_ENV}"

# shellcheck source=/dev/null
source "${BASE_CONFIG}"
[[ "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'invalid PLATFORM_PROFILE'
[[ "${RUNTIME_CLASS-}" == 'kata-qemu-nvidia-gpu-snp' ]] || die 'unexpected RUNTIME_CLASS'
PROFILE_DIR="${PLATFORM_WORK_ROOT:?PLATFORM_WORK_ROOT is required}/${PLATFORM_PROFILE}"
[[ "$(dirname -- "${APPROVAL_ENV}")" == "${PROFILE_DIR}" ]] \
    || die 'APPROVAL-ENV must be inside the selected platform profile'

KUBECONFIG_PATH="${KUBECONFIG_PATH:-/etc/kubernetes/admin.conf}"
KCTL=(kubectl --kubeconfig "${KUBECONFIG_PATH}")
if [[ ! -r "${KUBECONFIG_PATH}" ]]; then KCTL=(sudo kubectl --kubeconfig "${KUBECONFIG_PATH}"); fi
"${KCTL[@]}" get runtimeclass "${RUNTIME_CLASS}" >/dev/null
"${KCTL[@]}" get nodes >/dev/null

SNP_GUEST_VERSION='0.9.2'
SNP_GUEST_SHA256='9f008af82c37ad1d152d6b6f22fe4bf892d45f278d8b2500f69591f238905481'
SNP_GUEST_URL="https://github.com/virtee/snpguest/releases/download/v${SNP_GUEST_VERSION}/snpguest"
UBUNTU_AMD64_DIGEST='sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316'
COLLECTOR_DIR="${PROFILE_DIR}/rehearsal-collector-build"
[[ ! -e "${COLLECTOR_DIR}" ]] || die "refusing to overwrite: ${COLLECTOR_DIR}"
completed=0
NAMESPACE=''
REGISTRY_CONTAINER=''
cleanup() {
    if [[ -n "${NAMESPACE}" ]]; then
        "${KCTL[@]}" delete namespace "${NAMESPACE}" --ignore-not-found --wait=false \
            >/dev/null 2>&1 || true
    fi
    if [[ -n "${REGISTRY_CONTAINER}" ]]; then
        docker rm --force "${REGISTRY_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if (( completed == 0 )); then
        printf 'Retaining failed rehearsal inputs and evidence at %s for diagnosis.\n' "${COLLECTOR_DIR}" >&2
    fi
}
trap cleanup EXIT
install -d -m 0700 "${COLLECTOR_DIR}"
install -m 0644 "${SCRIPT_DIR}/rehearsal-collector/Dockerfile" "${COLLECTOR_DIR}/Dockerfile"
install -m 0755 "${SCRIPT_DIR}/rehearsal-collector/collect-snp-evidence.sh" \
    "${COLLECTOR_DIR}/collect-snp-evidence.sh"
grep -Fq "FROM ubuntu@${UBUNTU_AMD64_DIGEST}" "${COLLECTOR_DIR}/Dockerfile" \
    || die 'collector Dockerfile does not contain the approved amd64 base digest'
curl --fail --location --proto '=https' --tlsv1.2 \
    --output "${COLLECTOR_DIR}/snpguest" "${SNP_GUEST_URL}"
printf '%s  %s\n' "${SNP_GUEST_SHA256}" "${COLLECTOR_DIR}/snpguest" | sha256sum --check --strict
chmod 0755 "${COLLECTOR_DIR}/snpguest"

IMAGE="localhost/coco-snp-rehearsal:snpguest-${SNP_GUEST_VERSION}"
docker build --pull --tag "${IMAGE}" "${COLLECTOR_DIR}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
[[ "${IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]] || die 'collector image has no immutable image ID'

# Confidential Kata performs guest pull, so a host-only containerd import is
# intentionally insufficient. Publish the collector to a short-lived TLS
# registry reachable from inside the guest and carry its private CA in measured
# init-data. The registry is removed on every exit.
NODE_IP="$(ip -4 route get 1.1.1.1 | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
[[ "${NODE_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die 'could not derive NODE_IP'
REGISTRY_PORT=5443
REGISTRY_HOST="${NODE_IP}:${REGISTRY_PORT}"
REGISTRY_CONTAINER="coco-rehearsal-registry-${PLATFORM_PROFILE//[^a-zA-Z0-9_.-]/-}"
REGISTRY_IMAGE='docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373'
TLS_DIR="${COLLECTOR_DIR}/registry-tls"
SKOPEO_CERT_DIR="${COLLECTOR_DIR}/skopeo-certs"
REGISTRY_DATA="${COLLECTOR_DIR}/registry-data"
install -d -m 0700 "${TLS_DIR}" "${SKOPEO_CERT_DIR}" "${REGISTRY_DATA}"
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 2 \
    -keyout "${TLS_DIR}/ca.key" -out "${TLS_DIR}/ca.crt" \
    -subj '/CN=CoCo trusted rehearsal registry CA' >/dev/null 2>&1
openssl req -new -newkey rsa:3072 -nodes -sha256 \
    -keyout "${TLS_DIR}/server.key" -out "${TLS_DIR}/server.csr" \
    -subj "/CN=${NODE_IP}" >/dev/null 2>&1
printf 'subjectAltName=IP:%s\nextendedKeyUsage=serverAuth\n' "${NODE_IP}" > "${TLS_DIR}/server.ext"
openssl x509 -req -sha256 -days 2 -in "${TLS_DIR}/server.csr" \
    -CA "${TLS_DIR}/ca.crt" -CAkey "${TLS_DIR}/ca.key" -CAcreateserial \
    -extfile "${TLS_DIR}/server.ext" -out "${TLS_DIR}/server.crt" >/dev/null 2>&1
chmod 0600 "${TLS_DIR}/ca.key" "${TLS_DIR}/server.key"
chmod 0644 "${TLS_DIR}/ca.crt" "${TLS_DIR}/server.crt"
install -m 0644 "${TLS_DIR}/ca.crt" "${SKOPEO_CERT_DIR}/ca.crt"
docker run --detach --rm --name "${REGISTRY_CONTAINER}" --network host \
    --env "REGISTRY_HTTP_ADDR=${NODE_IP}:${REGISTRY_PORT}" \
    --env REGISTRY_HTTP_TLS_CERTIFICATE=/tls/server.crt \
    --env REGISTRY_HTTP_TLS_KEY=/tls/server.key \
    --volume "${TLS_DIR}:/tls:ro" --volume "${REGISTRY_DATA}:/var/lib/registry" \
    "${REGISTRY_IMAGE}" >/dev/null
for _ in {1..30}; do
    curl --fail --silent --show-error --cacert "${TLS_DIR}/ca.crt" \
        "https://${REGISTRY_HOST}/v2/" >/dev/null 2>&1 && break
    sleep 1
done
curl --fail --silent --show-error --cacert "${TLS_DIR}/ca.crt" \
    "https://${REGISTRY_HOST}/v2/" >/dev/null || die 'temporary TLS registry is not ready'
TAGGED_IMAGE="${REGISTRY_HOST}/coco-snp-rehearsal:snpguest-${SNP_GUEST_VERSION}"
skopeo copy --dest-cert-dir "${SKOPEO_CERT_DIR}" "docker-daemon:${IMAGE}" "docker://${TAGGED_IMAGE}" >/dev/null
IMAGE_DIGEST="$(skopeo inspect --cert-dir "${SKOPEO_CERT_DIR}" --format '{{.Digest}}' "docker://${TAGGED_IMAGE}")"
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die 'registry returned an invalid collector digest'
PUBLISHED_IMAGE="${REGISTRY_HOST}/coco-snp-rehearsal@${IMAGE_DIGEST}"

# Kubelet must see the tag locally without connecting to the private registry;
# after Kata starts, confidential-data-hub guest-pulls that same tag using the
# CA carried in measured init-data. The registry is immutable for this run and
# the independently resolved digest is retained in the evidence record.
docker tag "${IMAGE}" "${TAGGED_IMAGE}"
docker save --output "${COLLECTOR_DIR}/collector-host-cache.tar" "${TAGGED_IMAGE}"
sudo ctr --namespace k8s.io images import "${COLLECTOR_DIR}/collector-host-cache.tar" >/dev/null
POD_IMAGE="${TAGGED_IMAGE}"

openssl rand 64 > "${COLLECTOR_DIR}/request-data.bin"
CHALLENGE_SHA256="$(sha256sum "${COLLECTOR_DIR}/request-data.bin" | awk '{print $1}')"
RUN_ID="$(printf '%s' "${CHALLENGE_SHA256}" | cut -c1-16)"
NAMESPACE="coco-platform-rehearsal-${RUN_ID}"
POD="snp-evidence-${RUN_ID}"
LOG_FILE="${COLLECTOR_DIR}/collector.log"
POD_FILE="${COLLECTOR_DIR}/collector-pod.yaml"
INITDATA_FILE="${COLLECTOR_DIR}/initdata.toml"
REGISTRY_CA="$(<"${TLS_DIR}/ca.crt")"
cat > "${INITDATA_FILE}" <<EOF
version = "0.1.0"
algorithm = "sha256"

[data]
"cdh.toml" = '''
[kbc]
name = "offline_fs_kbc"
url = ""

[image]
# This collector is built and served entirely by the trusted rehearsal host.
# Do not configure an image-security policy here: CDH treats the field as
# optional, while the trusted workflow independently pins the source binary,
# base image, local image ID, registry digest, and ephemeral-registry TLS CA.
extra_root_certificates = ["""${REGISTRY_CA}"""]

[image.registry_config]
unqualified-search-registries = ["docker.io"]

[[image.registry_config.registry]]
location = "${REGISTRY_HOST}"
insecure = false
'''
EOF
INITDATA="$(gzip -n -c "${INITDATA_FILE}" | base64 -w0)"

"${KCTL[@]}" create namespace "${NAMESPACE}" >/dev/null
"${KCTL[@]}" -n "${NAMESPACE}" create configmap trusted-challenge \
    --from-file=request-data.bin="${COLLECTOR_DIR}/request-data.bin" >/dev/null
cat > "${POD_FILE}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  namespace: ${NAMESPACE}
  labels:
    coco.nvidia.com/purpose: trusted-platform-rehearsal
  annotations:
    io.katacontainers.config.hypervisor.cc_init_data: ${INITDATA}
spec:
  runtimeClassName: ${RUNTIME_CLASS}
  automountServiceAccountToken: false
  enableServiceLinks: false
  restartPolicy: Never
  containers:
  - name: collector
    image: ${POD_IMAGE}
    imagePullPolicy: Never
    stdin: false
    tty: false
    securityContext:
      privileged: true
      runAsUser: 0
      runAsGroup: 0
    volumeMounts:
    - name: challenge
      mountPath: /challenge
      readOnly: true
  volumes:
  - name: challenge
    configMap:
      name: trusted-challenge
      defaultMode: 0400
EOF
if [[ -n "${REHEARSAL_WORKLOAD_YAML:-}" ]]; then
    python3 - "${POD_FILE}" "${REHEARSAL_WORKLOAD_YAML}" <<'PY'
import sys
from pathlib import Path
import yaml
pod_path, source_path = map(Path, sys.argv[1:])
pod = yaml.safe_load(pod_path.read_text())
workload = yaml.safe_load(source_path.read_text())
spec = workload['spec']
if spec.get('runtimeClassName') != pod['spec']['runtimeClassName']:
    raise SystemExit('Workload RuntimeClass differs from the approved rehearsal')
if len(spec.get('containers', [])) != 1 or spec.get('initContainers'):
    raise SystemExit('This rehearsal currently supports one workload container and no init containers')
pod['spec']['containers'][0]['resources'] = spec['containers'][0].get('resources', {})
for name in ('hostNetwork', 'hostPID', 'hostIPC'):
    if spec.get(name, False):
        raise SystemExit(f'Unsupported workload launch setting: {name}')
for annotation in workload.get('metadata', {}).get('annotations', {}):
    if annotation.startswith('io.katacontainers.config.') and annotation != 'io.katacontainers.config.hypervisor.cc_init_data':
        raise SystemExit(f'Workload has an unreviewed runtime override: {annotation}')
pod_path.write_text(yaml.safe_dump(pod, sort_keys=False))
PY
fi
"${KCTL[@]}" apply -f "${POD_FILE}" >/dev/null

deadline=$((SECONDS + 600))
while (( SECONDS < deadline )); do
    phase="$("${KCTL[@]}" -n "${NAMESPACE}" get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "${phase}" == Running && -n "${REHEARSAL_WORKLOAD_YAML:-}" && ! -s "${COLLECTOR_DIR}/actual-launch.json" ]]; then
        if sudo python3 "${SCRIPT_DIR}/capture-running-launch.py" "${NAMESPACE}" "${POD}" \
            /opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml \
            "${COLLECTOR_DIR}/actual-launch.json"; then
            sudo chown "$(id -u):$(id -g)" "${COLLECTOR_DIR}/actual-launch.json"
        else
            capture_status=$?
            [[ $capture_status == 75 ]] || die 'Cannot capture the actual rehearsal launch'
        fi
    fi
    [[ "${phase}" == Succeeded ]] && break
    if [[ "${phase}" == Failed ]]; then
        "${KCTL[@]}" -n "${NAMESPACE}" describe pod "${POD}" >&2 || true
        "${KCTL[@]}" -n "${NAMESPACE}" logs "${POD}" >&2 || true
        die 'rehearsal collector Pod failed'
    fi
    sleep 2
done
[[ "${phase-}" == Succeeded ]] || die 'timed out waiting for rehearsal collector Pod'
if [[ -n "${REHEARSAL_WORKLOAD_YAML:-}" ]]; then
    [[ -s "${COLLECTOR_DIR}/actual-launch.json" ]] || die 'Actual VM launch was not captured'
fi
"${KCTL[@]}" -n "${NAMESPACE}" logs "${POD}" > "${LOG_FILE}"

mapfile -t payloads < <(sed -n 's/^COCO_SNP_EVIDENCE_V1=//p' "${LOG_FILE}")
(( ${#payloads[@]} == 1 )) || die "expected one evidence payload, found ${#payloads[@]}"
printf '%s' "${payloads[0]}" | base64 --decode > "${COLLECTOR_DIR}/evidence.tar.gz"
mapfile -t members < <(tar -tzf "${COLLECTOR_DIR}/evidence.tar.gz" | sort)
expected=(SHA256SUMS attestation-report.bin attestation-report.txt request-data.bin)
for required in "${expected[@]}"; do
    printf '%s\n' "${members[@]}" | grep -Fxq "${required}" || die "evidence archive omits ${required}"
done
if printf '%s\n' "${members[@]}" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then die 'unsafe evidence archive path'; fi
EVIDENCE_INPUT="${COLLECTOR_DIR}/evidence-input"
install -d -m 0700 "${EVIDENCE_INPUT}"
tar -xzf "${COLLECTOR_DIR}/evidence.tar.gz" -C "${EVIDENCE_INPUT}"
(
    cd "${EVIDENCE_INPUT}"
    sha256sum --check --strict SHA256SUMS
)
cmp -- "${COLLECTOR_DIR}/request-data.bin" "${EVIDENCE_INPUT}/request-data.bin" \
    || die 'collector returned a different challenge'

# Verify REPORT_DATA equals the exact fresh 64-byte challenge before accepting
# any evidence. REPORT_DATA occupies bytes 0x50..0x8f in the SNP ABI report.
python3 - "${EVIDENCE_INPUT}/attestation-report.bin" "${COLLECTOR_DIR}/request-data.bin" <<'PY'
import hmac
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_bytes()
challenge = Path(sys.argv[2]).read_bytes()
if len(report) < 0x90:
    raise SystemExit("attestation report is truncated")
if len(challenge) != 64 or not hmac.compare_digest(report[0x50:0x90], challenge):
    raise SystemExit("signed REPORT_DATA does not match the fresh trusted challenge")
PY

# Fetch the complete AMD chain on the trusted host. The minimal guest does not
# inherit the trusted host's outbound TLS trust configuration, and the chain is
# public verification material rather than confidential guest evidence.
install -d -m 0700 "${EVIDENCE_INPUT}/certs"
PATH="${COLLECTOR_DIR}:${PATH}" snpguest fetch ca pem "${EVIDENCE_INPUT}/certs" \
    --report "${EVIDENCE_INPUT}/attestation-report.bin" --endorser vcek
PATH="${COLLECTOR_DIR}:${PATH}" snpguest fetch vcek pem "${EVIDENCE_INPUT}/certs" \
    "${EVIDENCE_INPUT}/attestation-report.bin"

PATH="${COLLECTOR_DIR}:${PATH}" "${SCRIPT_DIR}/record-reported-tcb.sh" \
    "${BASE_CONFIG}" "${APPROVAL_ENV}" \
    "${EVIDENCE_INPUT}/attestation-report.bin" "${EVIDENCE_INPUT}/certs"

MEASUREMENT="$(python3 - "${EVIDENCE_INPUT}/attestation-report.bin" <<'PY'
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_bytes()
if len(report) < 0xc0:
    raise SystemExit("attestation report is truncated before MEASUREMENT")
print(report[0x90:0xc0].hex())
PY
)"
REHEARSAL_EVIDENCE="${PROFILE_DIR}/rehearsal-evidence.txt"
[[ ! -e "${REHEARSAL_EVIDENCE}" ]] || die "refusing to overwrite: ${REHEARSAL_EVIDENCE}"
cat > "${REHEARSAL_EVIDENCE}" <<EOF
Collector image ID: ${IMAGE_ID}
Collector registry reference: ${PUBLISHED_IMAGE}
Collector base image: ubuntu@${UBUNTU_AMD64_DIGEST}
snpguest version: ${SNP_GUEST_VERSION}
snpguest SHA-256: ${SNP_GUEST_SHA256}
RuntimeClass: ${RUNTIME_CLASS}
Pod manifest SHA-256: $(sha256sum "${POD_FILE}" | awk '{print $1}')
Challenge SHA-256: ${CHALLENGE_SHA256}
Attestation report SHA-256: $(sha256sum "${EVIDENCE_INPUT}/attestation-report.bin" | awk '{print $1}')
Reported launch measurement: ${MEASUREMENT}
EOF
if [[ -n "${REHEARSAL_WORKLOAD_YAML:-}" ]]; then
    printf 'Actual launch SHA-256: %s\n' "$(sha256sum "${COLLECTOR_DIR}/actual-launch.json" | awk '{print $1}')" >> "${REHEARSAL_EVIDENCE}"
    printf 'Workload source SHA-256: %s\n' "$(sha256sum "${REHEARSAL_WORKLOAD_YAML}" | awk '{print $1}')" >> "${REHEARSAL_EVIDENCE}"
fi
chmod 0600 "${REHEARSAL_EVIDENCE}"
python3 - "${APPROVAL_ENV}" "${MEASUREMENT}" <<'PY'
import os
import re
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
updates = {
    "REHEARSAL_EVIDENCE_FILE": "rehearsal-evidence.txt",
}
text = path.read_text()
for key, value in updates.items():
    text, count = re.subn(rf"(?m)^{key}=.*$", f'{key}="{value}"', text)
    if count != 1:
        raise SystemExit(f"expected exactly one {key} assignment; found {count}")
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY

printf 'Fail-closed in-cluster SNP rehearsal completed.\n'
printf '  collector image ID: %s\n' "${IMAGE_ID}"
printf '  collector registry digest: %s\n' "${PUBLISHED_IMAGE}"
printf '  challenge SHA-256: %s\n' "${CHALLENGE_SHA256}"
printf '  signed launch measurement: %s\n' "${MEASUREMENT}"
printf '  retained build/evidence: %s\n' "${COLLECTOR_DIR}"
completed=1
