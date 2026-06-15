#!/usr/bin/env bash

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo
  echo "==> $*"
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
  done
}

is_truthy() {
  case "${1:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Yy])
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

safe_name() {
  python3 - "$1" <<'PY'
import re
import sys

name = re.sub(r"[^a-z0-9-]", "-", sys.argv[1].lower())
name = re.sub(r"-+", "-", name).strip("-") or "site"
if not name[0].isalpha():
    name = f"site-{name}"
print(name[:63].rstrip("-"))
PY
}

json_data_field() {
  local field=$1
  python3 -c '
import json
import sys

field = sys.argv[1]
payload = json.load(sys.stdin)
data = payload.get("data") or {}
value = data
for part in field.split("."):
    value = value[part]
print(value)
' "$field"
}

normalize_job_id() {
  python3 - "$1" <<'PY'
import re
import sys

name = sys.argv[1].lower()
name = re.sub(r"[^a-z0-9-]", "", name)
if name and name[0].isdigit():
    name = "j" + name
print(name[:63].rstrip("-"))
PY
}

append_yaml_secret_list() {
  local file=$1
  local indent=$2
  local key=$3
  local values=$4
  local name

  [[ -n "${values}" ]] || return 0
  printf "%*s%s:\n" "${indent}" "" "${key}" >>"${file}"
  for name in ${values}; do
    printf "%*s- %s\n" "$((indent + 2))" "" "${name}" >>"${file}"
  done
}

append_k8s_image_pull_secrets() {
  local file=$1
  local indent=$2
  local values=$3
  local name

  [[ -n "${values}" ]] || return 0
  printf "%*simagePullSecrets:\n" "${indent}" "" >>"${file}"
  for name in ${values}; do
    printf "%*s- name: %s\n" "$((indent + 2))" "" "${name}" >>"${file}"
  done
}

init_k8s_env() {
  local require_image=${1:-false}

  SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

  KUBE_CMD="${KUBE_CMD:-oc}"
  PROJECT_NAME="${PROJECT_NAME:-openshift_nvflare_e2e}"
  NAMESPACE="${NAMESPACE:-nvflare-e2e}"
  SERVER_NAME="${SERVER_NAME:-nvflare-server}"
  SERVER_SERVICE_NAME="${SERVER_SERVICE_NAME:-${SERVER_NAME}}"
  SERVER_HOST="${SERVER_HOST:-${SERVER_SERVICE_NAME}}"
  CLIENTS="${CLIENTS:-site-1 site-2}"
  ADMIN_USER="${ADMIN_USER:-admin@nvidia.com}"
  ADMIN_ROLE="${ADMIN_ROLE:-lead}"
  ORG="${ORG:-nvidia}"
  FED_LEARN_PORT="${FED_LEARN_PORT:-8002}"
  ADMIN_PORT="${ADMIN_PORT:-8003}"
  PARENT_PORT="${PARENT_PORT:-8102}"
  WORKSPACE_MOUNT_PATH="${WORKSPACE_MOUNT_PATH:-/var/tmp/nvflare/workspace}"
  PARENT_PYTHON_PATH="${PARENT_PYTHON_PATH:-python}"
  PARENT_CPU="${PARENT_CPU:-}"
  PARENT_MEMORY="${PARENT_MEMORY:-}"
  ADMIN_PYTHON_PATH="${ADMIN_PYTHON_PATH:-${PARENT_PYTHON_PATH}}"
  JOB_PYTHON_PATH="${JOB_PYTHON_PATH:-/usr/local/bin/python3}"
  JOB_PENDING_TIMEOUT="${JOB_PENDING_TIMEOUT:-300}"
  WORKSPACE_STORAGE="${WORKSPACE_STORAGE:-2Gi}"
  WORKSPACE_STAGING_MODE="${WORKSPACE_STAGING_MODE:-pvc}"
  STORAGE_CLASS="${STORAGE_CLASS:-}"
  WORK_DIR="${WORK_DIR:-/tmp/nvflare/openshift-e2e}"
  CLEAN_WORK_DIR="${CLEAN_WORK_DIR:-false}"
  ALLOW_DELETE_OUTSIDE_TMP="${ALLOW_DELETE_OUTSIDE_TMP:-false}"
  DELETE_NAMESPACE_ON_EXIT="${DELETE_NAMESPACE_ON_EXIT:-false}"
  DELETE_ADMIN_POD_ON_EXIT="${DELETE_ADMIN_POD_ON_EXIT:-false}"
  POD_READY_TIMEOUT="${POD_READY_TIMEOUT:-180s}"
  ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
  JOB_WAIT_TIMEOUT="${JOB_WAIT_TIMEOUT:-900}"
  JOB_WAIT_INTERVAL="${JOB_WAIT_INTERVAL:-5}"
  JOB_POD_APPEAR_TIMEOUT="${JOB_POD_APPEAR_TIMEOUT:-180}"
  NVFLARE_CONNECT_TIMEOUT="${NVFLARE_CONNECT_TIMEOUT:-10}"
  NUM_ROUNDS="${NUM_ROUNDS:-1}"
  COPY_IMAGE="${COPY_IMAGE:-busybox:1.36}"
  PARENT_IMAGE_PULL_SECRETS="${PARENT_IMAGE_PULL_SECRETS:-}"
  JOB_IMAGE_PULL_SECRETS="${JOB_IMAGE_PULL_SECRETS:-${PARENT_IMAGE_PULL_SECRETS}}"
  JOB_CPU="${JOB_CPU:-}"
  JOB_MEMORY="${JOB_MEMORY:-}"
  JOB_EPHEMERAL_STORAGE="${JOB_EPHEMERAL_STORAGE:-1Gi}"
  SUBMIT_TOKEN="${SUBMIT_TOKEN:-openshift-e2e-$(date +%Y%m%d%H%M%S)}"

  if [[ "${require_image}" == "true" ]]; then
    : "${IMAGE:?Set IMAGE to a cluster-pullable NVFlare image before running this script.}"
  fi
  JOB_IMAGE="${JOB_IMAGE:-${IMAGE:-}}"
  ADMIN_IMAGE="${ADMIN_IMAGE:-${IMAGE:-}}"

  case "${WORKSPACE_STAGING_MODE}" in
    pvc|configmap-secret)
      ;;
    *)
      fail "WORKSPACE_STAGING_MODE must be 'pvc' or 'configmap-secret', got: ${WORKSPACE_STAGING_MODE}"
      ;;
  esac

  read -r -a CLIENT_ARRAY <<<"${CLIENTS}"
  CLIENT_COUNT="${#CLIENT_ARRAY[@]}"
  ((CLIENT_COUNT > 0)) || fail "CLIENTS must contain at least one client site"

  PARTICIPANTS=("${SERVER_NAME}" "${CLIENT_ARRAY[@]}" "${ADMIN_USER}")
  RUNTIME_PARTICIPANTS=("${SERVER_NAME}" "${CLIENT_ARRAY[@]}")

  PROJECT_FILE="${WORK_DIR}/project.yml"
  PACKAGE_WORKSPACE="${WORK_DIR}/workspace"
  PREPARED_DIR="${WORK_DIR}/prepared"
  PREPARE_CONFIG_DIR="${WORK_DIR}/prepare-configs"
  JOB_DIR="${WORK_DIR}/jobs/hello-numpy-k8s"
  ADMIN_POD="${ADMIN_POD:-nvflare-admin}"
  ADMIN_POD_FILE="${WORK_DIR}/admin-pod.yaml"
  PROD_DIR="${PACKAGE_WORKSPACE}/${PROJECT_NAME}/prod_00"
  LAST_JOB_ID_FILE="${WORK_DIR}/last_job_id"

  [[ -d "${REPO_ROOT}" ]] || fail "REPO_ROOT does not exist: ${REPO_ROOT}"
}

clean_work_dir_if_requested() {
  if [[ "${CLEAN_WORK_DIR}" != "true" ]]; then
    return
  fi

  case "${WORK_DIR}" in
    /tmp/nvflare/*)
      rm -rf "${WORK_DIR}"
      ;;
    *)
      [[ "${ALLOW_DELETE_OUTSIDE_TMP}" == "true" ]] || {
        fail "Refusing to delete WORK_DIR outside /tmp/nvflare: ${WORK_DIR}. Set ALLOW_DELETE_OUTSIDE_TMP=true to allow."
      }
      rm -rf "${WORK_DIR}"
      ;;
  esac
}

ensure_work_dirs() {
  mkdir -p "${PACKAGE_WORKSPACE}" "${PREPARED_DIR}" "${PREPARE_CONFIG_DIR}" "$(dirname "${JOB_DIR}")"
}

require_provisioned_workspace() {
  local participant

  [[ -d "${PROD_DIR}" ]] || fail "Provisioned prod dir not found: ${PROD_DIR}. Run k8s_provision.sh first."
  for participant in "${PARTICIPANTS[@]}"; do
    [[ -d "${PROD_DIR}/${participant}" ]] || fail "Missing provisioned participant folder: ${PROD_DIR}/${participant}"
  done
}

write_project_file() {
  local participant
  local host
  local seen_hosts=""

  cat >"${PROJECT_FILE}" <<EOF
api_version: 3
name: ${PROJECT_NAME}
description: NVFlare OpenShift e2e deployment

participants:
  - name: ${SERVER_NAME}
    type: server
    org: ${ORG}
    default_host: "${SERVER_HOST}"
    host_names:
EOF

  for host in "${SERVER_NAME}" "${SERVER_HOST}" "${SERVER_SERVICE_NAME}"; do
    [[ -n "${host}" ]] || continue
    case " ${seen_hosts} " in
      *" ${host} "*)
        continue
        ;;
    esac
    seen_hosts="${seen_hosts} ${host}"
    cat >>"${PROJECT_FILE}" <<EOF
      - "${host}"
EOF
  done

  cat >>"${PROJECT_FILE}" <<EOF
    fed_learn_port: ${FED_LEARN_PORT}
    admin_port: ${ADMIN_PORT}
EOF

  for participant in "${CLIENT_ARRAY[@]}"; do
    cat >>"${PROJECT_FILE}" <<EOF
  - name: ${participant}
    type: client
    org: ${ORG}
EOF
  done

  cat >>"${PROJECT_FILE}" <<EOF
  - name: ${ADMIN_USER}
    type: admin
    org: ${ORG}
    role: ${ADMIN_ROLE}

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file:
        - master_template.yml
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      scheme: grpc
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
EOF
}

write_prepare_config() {
  local participant=$1
  local pvc=$2
  local file="${PREPARE_CONFIG_DIR}/k8s-${participant}.yaml"

  cat >"${file}" <<EOF
runtime: k8s
namespace: ${NAMESPACE}
EOF
  if [[ "${SERVER_SERVICE_NAME}" != "nvflare-server" ]]; then
    cat >>"${file}" <<EOF
server_service_name: ${SERVER_SERVICE_NAME}
EOF
  fi
  cat >>"${file}" <<EOF
parent:
  docker_image: "${IMAGE}"
  parent_port: ${PARENT_PORT}
  workspace_pvc: ${pvc}
  workspace_mount_path: ${WORKSPACE_MOUNT_PATH}
EOF
  if [[ "${PARENT_PYTHON_PATH}" != "/usr/local/bin/python3" ]]; then
    cat >>"${file}" <<EOF
  python_path: ${PARENT_PYTHON_PATH}
EOF
  fi
  if [[ -n "${PARENT_CPU}" || -n "${PARENT_MEMORY}" ]]; then
    cat >>"${file}" <<EOF
  resources:
    requests:
EOF
    if [[ -n "${PARENT_CPU}" ]]; then
      cat >>"${file}" <<EOF
      cpu: "${PARENT_CPU}"
EOF
    fi
    if [[ -n "${PARENT_MEMORY}" ]]; then
      cat >>"${file}" <<EOF
      memory: "${PARENT_MEMORY}"
EOF
    fi
  fi
  append_yaml_secret_list "${file}" 2 "image_pull_secrets" "${PARENT_IMAGE_PULL_SECRETS}"

  cat >>"${file}" <<EOF
job_launcher:
  config_file_path:
  default_python_path: ${JOB_PYTHON_PATH}
  pending_timeout: ${JOB_PENDING_TIMEOUT}
EOF
  append_yaml_secret_list "${file}" 2 "image_pull_secrets" "${JOB_IMAGE_PULL_SECRETS}"

  echo "${file}"
}

verify_prepared_launcher() {
  local participant=$1
  local role=$2
  local resources_file="${PREPARED_DIR}/${participant}/local/resources.json.default"

  [[ -f "${resources_file}" ]] || fail "Missing prepared resources file: ${resources_file}"
  python3 - "${resources_file}" "${role}" "${NAMESPACE}" "${WORKSPACE_MOUNT_PATH}" <<'PY'
import json
import sys

resources_file, role, namespace, workspace_mount_path = sys.argv[1:5]
expected = {
    "server": "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher",
    "client": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
}[role]

with open(resources_file, "r", encoding="utf-8") as f:
    resources = json.load(f)

components = resources.get("components") or []
launcher = next((c for c in components if c.get("id") == "k8s_launcher"), None)
if not launcher:
    raise SystemExit(f"no k8s_launcher component found in {resources_file}")
if launcher.get("path") != expected:
    raise SystemExit(f"k8s_launcher path {launcher.get('path')!r}; expected {expected!r}")
args = launcher.get("args") or {}
if args.get("namespace") != namespace:
    raise SystemExit(f"k8s_launcher namespace {args.get('namespace')!r}; expected {namespace!r}")
config_file_path = args.get("config_file_path")
if config_file_path not in (None, ""):
    raise SystemExit(
        f"k8s_launcher config_file_path {config_file_path!r}; expected null/empty for in-cluster config"
    )
launcher_workspace_mount_path = args.get("workspace_mount_path", "/var/tmp/nvflare/workspace")
if launcher_workspace_mount_path != workspace_mount_path:
    raise SystemExit(
        f"k8s_launcher workspace_mount_path {launcher_workspace_mount_path!r}; expected {workspace_mount_path!r}"
    )
PY
}

normalize_prepared_resources() {
  local participant=$1
  local resources_file="${PREPARED_DIR}/${participant}/local/resources.json.default"

  python3 - "${resources_file}" <<'PY'
import json
import pathlib
import sys

resources_file = pathlib.Path(sys.argv[1])
with resources_file.open("r", encoding="utf-8") as f:
    resources = json.load(f)

changed = False
for component in resources.get("components") or []:
    if component.get("path") == "nvflare.app_common.logging.system_log_streamer.SystemLogStreamer":
        component["path"] = "nvflare.app_common.logging.site_log_streamer.SiteLogStreamer"
        changed = True

if changed:
    with resources_file.open("w", encoding="utf-8") as f:
        json.dump(resources, f, indent=4)
        f.write("\n")
PY
}

create_namespace() {
  if "${KUBE_CMD}" get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    return
  fi

  if [[ "${KUBE_CMD}" == "oc" ]]; then
    "${KUBE_CMD}" new-project "${NAMESPACE}" >/dev/null
  else
    "${KUBE_CMD}" create namespace "${NAMESPACE}" >/dev/null
  fi
}

write_pvc_manifest() {
  local file=$1
  local pvc=$2
  local storage_class_line=""

  if [[ -n "${STORAGE_CLASS}" ]]; then
    storage_class_line="  storageClassName: ${STORAGE_CLASS}"
  fi

  cat >"${file}" <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${pvc}
spec:
${storage_class_line}
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: ${WORKSPACE_STORAGE}
EOF
}

wait_for_pvc_bound() {
  local pvc=$1
  local timeout_seconds=$2
  local phase=""
  local i

  for ((i = 0; i < timeout_seconds; i++)); do
    phase="$("${KUBE_CMD}" -n "${NAMESPACE}" get pvc "${pvc}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "${phase}" == "Bound" ]]; then
      return 0
    fi
    sleep 1
  done

  "${KUBE_CMD}" -n "${NAMESPACE}" describe pvc "${pvc}" || true
  fail "PVC ${pvc} did not become Bound; last phase=${phase:-unknown}"
}

write_copy_pod_manifest() {
  local file=$1
  local pod=$2
  local pvc=$3

  cat >"${file}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
spec:
  restartPolicy: Never
  containers:
    - name: copy
      image: ${COPY_IMAGE}
      command:
        - sh
        - -c
        - sleep 3600
      volumeMounts:
        - name: workspace
          mountPath: /mnt/nvflare-workspace
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: ${pvc}
EOF
  append_k8s_image_pull_secrets "${file}" 2 "${PARENT_IMAGE_PULL_SECRETS}"
}

stage_workspace_pvc() {
  local participant=$1
  local pvc=$2
  local safe
  local pod
  local pod_file

  safe="$(safe_name "${participant}")"
  pod="nvflare-copy-${safe}"
  pod_file="${WORK_DIR}/copy-${safe}.yaml"

  write_copy_pod_manifest "${pod_file}" "${pod}" "${pvc}"
  "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "${pod}" --ignore-not-found=true >/dev/null
  "${KUBE_CMD}" -n "${NAMESPACE}" apply -f "${pod_file}" >/dev/null
  wait_for_pvc_bound "${pvc}" 180
  "${KUBE_CMD}" -n "${NAMESPACE}" wait --for=condition=Ready "pod/${pod}" --timeout="${POD_READY_TIMEOUT}"

  "${KUBE_CMD}" -n "${NAMESPACE}" exec "${pod}" -- rm -rf /mnt/nvflare-workspace/startup /mnt/nvflare-workspace/local
  "${KUBE_CMD}" -n "${NAMESPACE}" exec "${pod}" -- mkdir -p /mnt/nvflare-workspace/startup /mnt/nvflare-workspace/local
  "${KUBE_CMD}" -n "${NAMESPACE}" cp "${PREPARED_DIR}/${participant}/startup/." "${pod}:/mnt/nvflare-workspace/startup"
  "${KUBE_CMD}" -n "${NAMESPACE}" cp "${PREPARED_DIR}/${participant}/local/." "${pod}:/mnt/nvflare-workspace/local"
  "${KUBE_CMD}" -n "${NAMESPACE}" exec "${pod}" -- ls -la /mnt/nvflare-workspace/startup /mnt/nvflare-workspace/local
  "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "${pod}" --ignore-not-found=true >/dev/null
}

stage_workspace_configmap_secret() {
  local participant=$1
  local kit="${PREPARED_DIR}/${participant}"

  nvflare deploy k8s stage "${kit}" --namespace "${NAMESPACE}" --kubectl "${KUBE_CMD}"
}

install_chart() {
  local participant=$1
  local deployment_existed=false

  if "${KUBE_CMD}" -n "${NAMESPACE}" get "deployment/${participant}" >/dev/null 2>&1; then
    deployment_existed=true
  fi
  helm upgrade --install "${participant}" "${PREPARED_DIR}/${participant}/helm_chart" --namespace "${NAMESPACE}"
  if [[ "${deployment_existed}" == "true" ]]; then
    "${KUBE_CMD}" -n "${NAMESPACE}" rollout restart "deployment/${participant}"
  fi
  "${KUBE_CMD}" -n "${NAMESPACE}" rollout status "deployment/${participant}" --timeout="${ROLLOUT_TIMEOUT}"
}

verify_runtime_deployments() {
  local participant

  for participant in "${RUNTIME_PARTICIPANTS[@]}"; do
    "${KUBE_CMD}" -n "${NAMESPACE}" rollout status "deployment/${participant}" --timeout="${ROLLOUT_TIMEOUT}"
  done
}

verify_parent_kubernetes_client() {
  local participant=$1

  "${KUBE_CMD}" -n "${NAMESPACE}" exec "deploy/${participant}" -- "${PARENT_PYTHON_PATH}" -c '
import kubernetes

print(f"kubernetes-python-client={kubernetes.__version__}")
'
}

export_hello_numpy_job() {
  local job_dir=$1
  local job_parent

  job_parent="$(dirname "${job_dir}")"
  rm -rf "${job_dir}"
  mkdir -p "${job_parent}"
  python3 - "${REPO_ROOT}" "${job_parent}" "${CLIENT_COUNT}" "${NUM_ROUNDS}" <<'PY'
import os
import pathlib
import sys

repo_root, job_parent, client_count, num_rounds = sys.argv[1:5]
example_dir = os.path.join(repo_root, "examples", "hello-world", "hello-numpy")
sys.path.insert(0, repo_root)
os.chdir(example_dir)

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType

recipe = NumpyFedAvgRecipe(
    name="hello-numpy-k8s",
    min_clients=int(client_count),
    num_rounds=int(num_rounds),
    model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    train_script="client.py",
    train_args="--update_type full",
    launch_external_process=False,
    params_transfer_type=TransferType.FULL,
)
recipe.export(job_parent)
job_dir = pathlib.Path(job_parent) / "hello-numpy-k8s"
if not (job_dir / "meta.json").is_file():
    raise SystemExit(f"expected exported job at {job_dir}")
PY
}

patch_job_launcher_spec() {
  local job_dir=$1

  python3 - "${job_dir}" "${JOB_IMAGE}" "${JOB_PYTHON_PATH}" "${JOB_CPU}" "${JOB_MEMORY}" "${JOB_EPHEMERAL_STORAGE}" <<'PY'
import json
import pathlib
import sys

job_dir, image, python_path, cpu, memory, ephemeral_storage = sys.argv[1:7]
meta_path = pathlib.Path(job_dir) / "meta.json"
with meta_path.open("r", encoding="utf-8") as f:
    meta = json.load(f)

k8s_spec = {
    "image": image,
    "python_path": python_path,
    "ephemeral_storage": ephemeral_storage,
}
if cpu:
    k8s_spec["cpu"] = cpu
if memory:
    k8s_spec["memory"] = memory

meta.setdefault("launcher_spec", {})
meta["launcher_spec"].setdefault("default", {})
meta["launcher_spec"]["default"]["k8s"] = k8s_spec

with meta_path.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY
}

write_admin_pod_manifest() {
  cat >"${ADMIN_POD_FILE}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${ADMIN_POD}
spec:
  restartPolicy: Never
  containers:
    - name: admin
      image: ${ADMIN_IMAGE}
      command:
        - ${ADMIN_PYTHON_PATH}
        - -c
        - "import time; time.sleep(3600)"
      volumeMounts:
        - name: admin-work
          mountPath: /workspace
  volumes:
    - name: admin-work
      emptyDir: {}
EOF
  append_k8s_image_pull_secrets "${ADMIN_POD_FILE}" 2 "${PARENT_IMAGE_PULL_SECRETS}"
}

copy_dir_to_admin_pod() {
  local src=$1
  local dest=$2

  [[ -d "${src}" ]] || fail "Directory not found: ${src}"
  tar -C "${src}" -cf - . | "${KUBE_CMD}" -n "${NAMESPACE}" exec -i "${ADMIN_POD}" -- "${ADMIN_PYTHON_PATH}" -c '
import pathlib
import sys
import tarfile

dest = pathlib.Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
dest_root = dest.resolve()

with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for member in archive:
        target = (dest / member.name).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise RuntimeError(f"Refusing unsafe tar member: {member.name}")
        archive.extract(member, dest, set_attrs=False)
' "${dest}"
}

admin_pod_file_exists() {
  local path=$1

  "${KUBE_CMD}" -n "${NAMESPACE}" exec "${ADMIN_POD}" -- "${ADMIN_PYTHON_PATH}" -c '
import pathlib
import sys

raise SystemExit(0 if pathlib.Path(sys.argv[1]).is_file() else 1)
' "${path}"
}

prepare_admin_pod() {
  "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "${ADMIN_POD}" --ignore-not-found=true >/dev/null
  write_admin_pod_manifest
  "${KUBE_CMD}" -n "${NAMESPACE}" apply -f "${ADMIN_POD_FILE}" >/dev/null
  "${KUBE_CMD}" -n "${NAMESPACE}" wait --for=condition=Ready "pod/${ADMIN_POD}" --timeout="${POD_READY_TIMEOUT}"
  copy_dir_to_admin_pod "${PROD_DIR}/${ADMIN_USER}" /workspace/admin
  copy_dir_to_admin_pod "${JOB_DIR}" /workspace/job
  admin_pod_file_exists /workspace/admin/startup/fed_admin.json
  admin_pod_file_exists /workspace/job/meta.json
}

wait_for_job_pods() {
  local normalized_job_id=$1
  local min_count=$2
  local timeout_seconds=$3
  local count
  local i

  for ((i = 0; i < timeout_seconds; i++)); do
    count="$("${KUBE_CMD}" -n "${NAMESPACE}" get pods -o json | python3 -c '
import json
import sys

prefix = sys.argv[1]
items = json.load(sys.stdin).get("items") or []
matches = [item for item in items if item.get("metadata", {}).get("name", "").startswith(prefix)]
print(len(matches))
' "${normalized_job_id}")"
    if ((count >= min_count)); then
      "${KUBE_CMD}" -n "${NAMESPACE}" get pods | grep "^${normalized_job_id}" || true
      return 0
    fi
    sleep 1
  done

  "${KUBE_CMD}" -n "${NAMESPACE}" get pods
  fail "Expected at least ${min_count} K8s launcher job pods with prefix ${normalized_job_id}; found ${count:-0}"
}

submit_and_wait_for_job() {
  local submit_out
  local wait_out
  local job_id
  local normalized_job_id
  local min_job_pods
  local job_status

  submit_out="$("${KUBE_CMD}" -n "${NAMESPACE}" exec "${ADMIN_POD}" -- \
    "${ADMIN_PYTHON_PATH}" -m nvflare.cli --format json --connect-timeout "${NVFLARE_CONNECT_TIMEOUT}" \
    job submit -j /workspace/job --startup-kit /workspace/admin --submit-token "${SUBMIT_TOKEN}")"
  job_id="$(printf '%s' "${submit_out}" | json_data_field job_id)"
  [[ -n "${job_id}" ]] || fail "Job submission did not return a job_id: ${submit_out}"
  mkdir -p "$(dirname "${LAST_JOB_ID_FILE}")"
  printf "%s\n" "${job_id}" >"${LAST_JOB_ID_FILE}"
  echo "Submitted job_id=${job_id}"

  normalized_job_id="$(normalize_job_id "${job_id}")"
  min_job_pods="${MIN_JOB_PODS:-$((CLIENT_COUNT + 1))}"
  wait_for_job_pods "${normalized_job_id}" "${min_job_pods}" "${JOB_POD_APPEAR_TIMEOUT}"

  wait_out="$("${KUBE_CMD}" -n "${NAMESPACE}" exec "${ADMIN_POD}" -- \
    "${ADMIN_PYTHON_PATH}" -m nvflare.cli --format json --connect-timeout "${NVFLARE_CONNECT_TIMEOUT}" \
    job wait "${job_id}" --startup-kit /workspace/admin --timeout "${JOB_WAIT_TIMEOUT}" --interval "${JOB_WAIT_INTERVAL}")"
  echo "${wait_out}"
  job_status="$(printf '%s' "${wait_out}" | json_data_field status)"
  [[ "${job_status}" == "FINISHED:COMPLETED" ]] || fail "Job ${job_id} finished with status ${job_status}"
}

report_missing_pieces() {
  local copy_image_note=""

  if [[ "${WORKSPACE_STAGING_MODE}" == "pvc" ]]; then
    copy_image_note="  - A pullable COPY_IMAGE with sh, sleep, and tar installed. The deploy phase
    uses ${KUBE_CMD} cp to stage prepared startup files into workspace PVCs."
  fi

  cat <<EOF

Missing pieces these scripts cannot create for the cluster:
  - A pullable parent IMAGE with NVFlare, its K8S extra/Kubernetes Python
    client, and the Python executable named by PARENT_PYTHON_PATH installed.
${copy_image_note}
  - A pullable ADMIN_IMAGE with NVFlare and the Python executable named by
    ADMIN_PYTHON_PATH installed. The submit phase can use the parent IMAGE as
    ADMIN_IMAGE.
  - A pullable JOB_IMAGE with NVFlare, Python, numpy, and the runtime tools
    needed by submitted job pods.
  - Registry pull secrets, if the image is private. Set
    PARENT_IMAGE_PULL_SECRETS and JOB_IMAGE_PULL_SECRETS to existing Secret names.
  - A working StorageClass/PV provisioner. Set STORAGE_CLASS if the cluster has
    no default StorageClass.
  - OpenShift SCC compatibility for the image. If pods fail with security
    context errors, use an image that runs under restricted-v2 or bind an
    appropriate SCC to the generated service accounts.
  - External exposure. The submit script submits from an in-cluster admin pod
    because SERVER_HOST defaults to the in-namespace Service ${SERVER_HOST}.
EOF
}

cleanup_on_exit() {
  if is_truthy "${DELETE_NAMESPACE_ON_EXIT}"; then
    "${KUBE_CMD}" delete namespace "${NAMESPACE}" --ignore-not-found=true >/dev/null 2>&1 || true
  elif is_truthy "${DELETE_ADMIN_POD_ON_EXIT}"; then
    "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "${ADMIN_POD}" --ignore-not-found=true >/dev/null 2>&1 || true
  fi
}

run_provision_phase() {
  require_cmd nvflare python3
  clean_work_dir_if_requested
  ensure_work_dirs

  info "Writing nvflare provision project file"
  write_project_file

  info "Running nvflare provision"
  nvflare provision -p "${PROJECT_FILE}" -w "${PACKAGE_WORKSPACE}" --force
  [[ -d "${PROD_DIR}" ]] || fail "Expected packaged prod dir not found: ${PROD_DIR}"

  info "Provisioned startup kits"
  for participant_dir in "${PROD_DIR}"/*; do
    [[ -d "${participant_dir}" ]] || continue
    basename "${participant_dir}"
  done | sort
}

run_deploy_phase() {
  local participant
  local pvc
  local cfg
  local pvc_file

  require_cmd "${KUBE_CMD}" helm nvflare python3
  if [[ "${WORKSPACE_STAGING_MODE}" == "pvc" ]]; then
    require_cmd tar
  fi
  require_provisioned_workspace
  ensure_work_dirs

  info "Preparing K8s deployment kits with the built-in K8s launcher"
  for participant in "${RUNTIME_PARTICIPANTS[@]}"; do
    pvc="nvflare-ws-$(safe_name "${participant}")"
    cfg="$(write_prepare_config "${participant}" "${pvc}")"
    nvflare deploy prepare "${PROD_DIR}/${participant}" --output "${PREPARED_DIR}/${participant}" --config "${cfg}"
    normalize_prepared_resources "${participant}"
  done
  verify_prepared_launcher "${SERVER_NAME}" server
  for participant in "${CLIENT_ARRAY[@]}"; do
    verify_prepared_launcher "${participant}" client
  done

  info "Creating namespace and staging workspaces with mode ${WORKSPACE_STAGING_MODE}"
  create_namespace
  "${KUBE_CMD}" -n "${NAMESPACE}" get serviceaccount default >/dev/null
  for participant in "${RUNTIME_PARTICIPANTS[@]}"; do
    pvc="nvflare-ws-$(safe_name "${participant}")"
    pvc_file="${WORK_DIR}/pvc-${participant}.yaml"
    write_pvc_manifest "${pvc_file}" "${pvc}"
    "${KUBE_CMD}" -n "${NAMESPACE}" apply -f "${pvc_file}"
    if [[ "${WORKSPACE_STAGING_MODE}" == "pvc" ]]; then
      stage_workspace_pvc "${participant}" "${pvc}"
    else
      stage_workspace_configmap_secret "${participant}"
    fi
  done

  info "Installing generated Helm charts"
  install_chart "${SERVER_NAME}"
  for participant in "${CLIENT_ARRAY[@]}"; do
    install_chart "${participant}"
  done
  verify_runtime_deployments
  info "Verifying parent images include the Kubernetes Python client"
  for participant in "${RUNTIME_PARTICIPANTS[@]}"; do
    verify_parent_kubernetes_client "${participant}"
  done
  "${KUBE_CMD}" -n "${NAMESPACE}" get pods,svc,pvc
}

run_submit_job_phase() {
  require_cmd "${KUBE_CMD}" nvflare python3 tar
  require_provisioned_workspace
  [[ -n "${ADMIN_IMAGE}" ]] || fail "Set IMAGE or ADMIN_IMAGE before running this script."
  [[ -n "${JOB_IMAGE}" ]] || fail "Set IMAGE or JOB_IMAGE before running this script."
  verify_runtime_deployments

  info "Exporting hello-numpy job and configuring launcher_spec.default.k8s"
  export_hello_numpy_job "${JOB_DIR}"
  patch_job_launcher_spec "${JOB_DIR}"
  python3 -m json.tool "${JOB_DIR}/meta.json" >/dev/null

  info "Submitting job from an in-cluster admin pod"
  prepare_admin_pod
  submit_and_wait_for_job

  info "Job workflow completed successfully"
  "${KUBE_CMD}" -n "${NAMESPACE}" get pods
  report_missing_pieces
}
