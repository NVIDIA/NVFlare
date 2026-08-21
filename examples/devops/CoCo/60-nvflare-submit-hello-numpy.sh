#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Export, submit, and verify the standard NVFlare hello-numpy job against the
CoCo server/client deployment created by 50-nvflare-deploy.sh.

Usage:
  ./60-nvflare-submit-hello-numpy.sh

Run 40-nvflare-build-sign-images.sh and 50-nvflare-deploy.sh first. Stage 60
uses a loopback-only host tunnel to the server administration endpoint, waits
for FINISHED:COMPLETED, and prints bounded server/client job logs.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || {
  usage >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
need kubectl
need jq
need stat

NVFLARE_ADMIN_LOCAL_PORT="${NVFLARE_ADMIN_LOCAL_PORT:-18003}"
NVFLARE_CLI_CONNECT_TIMEOUT="${NVFLARE_CLI_CONNECT_TIMEOUT:-30}"
NVFLARE_JOB_WAIT_TIMEOUT="${NVFLARE_JOB_WAIT_TIMEOUT:-900}"
NVFLARE_JOB_WAIT_INTERVAL="${NVFLARE_JOB_WAIT_INTERVAL:-2}"
for value_name in NVFLARE_ADMIN_LOCAL_PORT NVFLARE_CLI_CONNECT_TIMEOUT NVFLARE_JOB_WAIT_TIMEOUT \
  NVFLARE_JOB_WAIT_INTERVAL; do
  [[ "${!value_name:-}" =~ ^[1-9][0-9]*$ ]] || die "$value_name must be a positive integer"
done
((NVFLARE_ADMIN_LOCAL_PORT >= 1024 && NVFLARE_ADMIN_LOCAL_PORT <= 65535)) ||
  die "NVFLARE_ADMIN_LOCAL_PORT must be between 1024 and 65535"

nvflare_cli="${NVFLARE_CLI:-}"
if [[ -z "$nvflare_cli" ]] && command -v nvflare >/dev/null 2>&1; then
  nvflare_cli="$(command -v nvflare)"
fi
if [[ -z "$nvflare_cli" && -x "$REPO_ROOT/.venv/bin/nvflare" ]]; then
  nvflare_cli="$REPO_ROOT/.venv/bin/nvflare"
fi
[[ -x "$nvflare_cli" ]] ||
  die "NVFlare CLI not found; install this checkout in a virtual environment or set NVFLARE_CLI"

nvflare_python="${NVFLARE_PYTHON:-}"
if [[ -z "$nvflare_python" && -x "$(dirname "$nvflare_cli")/python" ]]; then
  nvflare_python="$(dirname "$nvflare_cli")/python"
fi
if [[ -z "$nvflare_python" ]] && command -v python3 >/dev/null 2>&1; then
  nvflare_python="$(command -v python3)"
fi
[[ -x "$nvflare_python" ]] || die "NVFlare Python interpreter not found; set NVFLARE_PYTHON"
"$nvflare_python" -c \
  "import numpy, nvflare, tensorboard; assert tensorboard.__version__ == '$TENSORBOARD_VERSION'" ||
  die "NVFLARE_PYTHON must contain NVFlare, NumPy, and TensorBoard $TENSORBOARD_VERSION"

deployment_state="$STATE_DIR/nvflare-deployment/current.env"
[[ -r "$deployment_state" ]] ||
  die "Run 50-nvflare-deploy.sh first: active deployment state is missing at $deployment_state"
[[ "$(stat -c '%a' "$deployment_state")" == 600 ]] ||
  die "Active deployment state must have mode 0600: $deployment_state"
# This file is written by Stage 50 with shell-escaped values and mode 0600.
# shellcheck disable=SC1090
source "$deployment_state"
[[ "${DEPLOYED_NVFLARE_NAMESPACE:-}" == "$NVFLARE_NAMESPACE" ]] ||
  die "Stage-50 state namespace does not match NVFLARE_NAMESPACE; rerun Stage 50"
[[ "${DEPLOYED_NVFLARE_SERVER_NAME:-}" == "$NVFLARE_SERVER_NAME" ]] ||
  die "Stage-50 state server does not match NVFLARE_SERVER_NAME; rerun Stage 50"
[[ "${DEPLOYED_NVFLARE_CLIENT_NAME:-}" == "$NVFLARE_CLIENT_NAME" ]] ||
  die "Stage-50 state client does not match NVFLARE_CLIENT_NAME; rerun Stage 50"
admin_startup_kit="${DEPLOYED_NVFLARE_ADMIN_STARTUP_KIT:-}"
[[ -n "$admin_startup_kit" && -r "$admin_startup_kit/fed_admin.json" ]] ||
  die "The active administrator startup kit is missing; rerun Stage 50"

kctl -n "$NVFLARE_NAMESPACE" rollout status "deployment/$NVFLARE_SERVER_NAME" --timeout=2m
kctl -n "$NVFLARE_NAMESPACE" rollout status "deployment/$NVFLARE_CLIENT_NAME" --timeout=2m
server_image="$(kctl -n "$NVFLARE_NAMESPACE" get deployment "$NVFLARE_SERVER_NAME" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
client_image="$(kctl -n "$NVFLARE_NAMESPACE" get deployment "$NVFLARE_CLIENT_NAME" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
[[ "$server_image" == "${DEPLOYED_NVFLARE_SERVER_IMAGE:-}" ]] ||
  die "Running server image differs from the successful Stage-50 deployment; rerun Stage 50"
[[ "$client_image" == "${DEPLOYED_NVFLARE_CLIENT_IMAGE:-}" ]] ||
  die "Running client image differs from the successful Stage-50 deployment; rerun Stage 50"

hello_numpy_dir="$REPO_ROOT/examples/hello-world/hello-numpy"
[[ -r "$hello_numpy_dir/client.py" ]] || die "hello-numpy example is missing: $hello_numpy_dir"
runs_dir="$STATE_DIR/nvflare-job-runs"
mkdir -p "$runs_dir"
run_dir="$(mktemp -d "$runs_dir/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
job_parent="$run_dir/job-config"
mkdir -p "$job_parent"

log "Exporting the standard one-client, three-round hello-numpy recipe"
(
  cd "$hello_numpy_dir"
  "$nvflare_python" - "$job_parent" <<'PY'
import sys

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import add_experiment_tracking

export_dir = sys.argv[1]
recipe = NumpyFedAvgRecipe(
    name="hello-numpy",
    min_clients=1,
    num_rounds=3,
    model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    train_script="client.py",
    train_args="--update_type full",
    launch_external_process=False,
    params_transfer_type=TransferType.FULL,
)
add_experiment_tracking(recipe, tracking_type="tensorboard")
recipe.export(export_dir)
PY
)
job_dir="$job_parent/hello-numpy"
[[ -s "$job_dir/meta.json" ]] || die "hello-numpy export did not create $job_dir/meta.json"
[[ -s "$job_dir/app/config/config_fed_server.json" ]] ||
  die "hello-numpy export did not create the server job configuration"
client_config="$job_dir/app/config/config_fed_client.json"
[[ -s "$client_config" && -f "$job_dir/app/custom/client.py" ]] ||
  die "hello-numpy export did not create the expected client job definition"
fixed_executor=nvflare.app_opt.confidential_computing.coco_hello_numpy_executor.CoCoHelloNumpyExecutor
client_config_tmp="$client_config.$$"
jq --arg executor "$fixed_executor" '
  if ([.executors[] | select(.executor.args.task_script_path? == "client.py")] | length) != 1 then
    error("unexpected hello-numpy executor configuration")
  else
    (.executors[] | select(.executor.args.task_script_path? == "client.py") |
      .executor) = {"path": $executor, "args": {}}
  end
' "$client_config" >"$client_config_tmp" ||
  die "Could not bind hello-numpy to the trainer in the signed parent image"
mv "$client_config_tmp" "$client_config"
rm -f -- "$job_dir/app/custom/client.py"
rmdir "$job_dir/app/custom" ||
  die "Refusing to submit hello-numpy because custom/BYOC content remains"

admin_tmp="$(mktemp -d)"
admin_tunnel_pid=""
cleanup() {
  if [[ -n "$admin_tunnel_pid" ]]; then
    kill "$admin_tunnel_pid" 2>/dev/null || true
    wait "$admin_tunnel_pid" 2>/dev/null || true
  fi
  if [[ -n "${admin_tmp:-}" && -d "$admin_tmp" ]]; then
    rm -rf -- "$admin_tmp"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

admin_source_dir="$(dirname "$admin_startup_kit")"
[[ -d "$admin_source_dir/local" ]] ||
  die "The administrator participant kit is missing its local directory: $admin_source_dir/local"
cp -a "$admin_source_dir/." "$admin_tmp/"
# The CLI accepts either the participant directory or its startup child, then
# normalizes it to a workspace containing both startup/ and local/.
admin_kit="$admin_tmp"
admin_config="$admin_kit/startup/fed_admin.json"
server_identity="$(jq -er '.admin.server_identity | strings | select(length > 0)' "$admin_config")"
[[ "$server_identity" == "$NVFLARE_SERVER_NAME" ]] ||
  die "Administrator kit server identity is $server_identity instead of $NVFLARE_SERVER_NAME"
admin_config_tmp="$admin_tmp/.fed_admin.json.$$"
jq --arg host 127.0.0.1 --argjson port "$NVFLARE_ADMIN_LOCAL_PORT" \
  '.admin.host = $host | .admin.port = $port' "$admin_config" >"$admin_config_tmp"
mv "$admin_config_tmp" "$admin_config"

server_pod_ip="$(kctl -n "$NVFLARE_NAMESPACE" get pod -l "app=$NVFLARE_SERVER_NAME" \
  -o jsonpath='{.items[0].status.podIP}')"
[[ "$server_pod_ip" =~ ^[0-9a-fA-F:.]+$ ]] ||
  die "Could not resolve the NVFlare server Pod IP"
admin_tunnel_log="$run_dir/admin-tunnel.log"
log "Opening a loopback-only administration tunnel on port $NVFLARE_ADMIN_LOCAL_PORT"
# kubectl port-forward dials localhost in the host-side Pod network namespace.
# For a Kata Pod the server listener is inside the VM instead, so proxy from
# host loopback to the routable Pod IP without exposing another host endpoint.
"$nvflare_python" - "$NVFLARE_ADMIN_LOCAL_PORT" "$server_pod_ip" \
  >"$admin_tunnel_log" 2>&1 <<'PY' &
import socket
import socketserver
import sys
import threading

listen_port = int(sys.argv[1])
target = (sys.argv[2], 8003)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection(target, timeout=10) as upstream:
            def relay(source, destination):
                try:
                    while data := source.recv(65536):
                        destination.sendall(data)
                finally:
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass

            outgoing = threading.Thread(target=relay, args=(self.request, upstream), daemon=True)
            outgoing.start()
            relay(upstream, self.request)
            outgoing.join(timeout=1)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with Server(("127.0.0.1", listen_port), Handler) as server:
    print(f"Forwarding 127.0.0.1:{listen_port} to {target[0]}:{target[1]}", flush=True)
    server.serve_forever()
PY
admin_tunnel_pid=$!

admin_tunnel_ready() {
  kill -0 "$admin_tunnel_pid" 2>/dev/null || return 1
  "$nvflare_python" - "$NVFLARE_ADMIN_LOCAL_PORT" 2>/dev/null <<'PY'
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
}
admin_tunnel_deadline=$((SECONDS + 30))
until admin_tunnel_ready; do
  if ((SECONDS >= admin_tunnel_deadline)) || ! kill -0 "$admin_tunnel_pid" 2>/dev/null; then
    sed -n '1,120p' "$admin_tunnel_log" >&2
    die "Could not establish the NVFlare administration tunnel"
  fi
  sleep 1
done
if ! kill -0 "$admin_tunnel_pid" 2>/dev/null; then
  sed -n '1,120p' "$admin_tunnel_log" >&2
  die "NVFlare administration tunnel exited unexpectedly"
fi

log "Submitting hello-numpy through the mTLS administrator startup kit"
submit_stderr="$run_dir/submit.stderr"
submit_rc=0
submit_json="$("$nvflare_cli" --format json --connect-timeout "$NVFLARE_CLI_CONNECT_TIMEOUT" \
  job submit -j "$job_dir" --startup-kit "$admin_kit" 2> >(tee "$submit_stderr" >&2))" || submit_rc=$?
printf '%s\n' "$submit_json" >"$run_dir/submit.json"
((submit_rc == 0)) || die "hello-numpy submission failed; see $run_dir/submit.json and $submit_stderr"
job_id="$(jq -er 'select(.status == "ok") | .data.job_id | strings | select(length > 0)' \
  <<<"$submit_json")" || die "NVFlare submission did not return a job ID"
echo "Submitted job ID: $job_id"

log "Waiting for hello-numpy to finish"
wait_stderr="$run_dir/wait.stderr"
wait_rc=0
wait_json="$("$nvflare_cli" --format json --connect-timeout "$NVFLARE_CLI_CONNECT_TIMEOUT" \
  job wait "$job_id" --startup-kit "$admin_kit" --timeout "$NVFLARE_JOB_WAIT_TIMEOUT" \
  --interval "$NVFLARE_JOB_WAIT_INTERVAL" 2> >(tee "$wait_stderr" >&2))" || wait_rc=$?
printf '%s\n' "$wait_json" >"$run_dir/wait.json"
if jq -e . >/dev/null 2>&1 <<<"$wait_json"; then
  jq . <<<"$wait_json"
else
  printf '%s\n' "$wait_json"
fi

log "Showing the final bounded server/client job logs"
if ! "$nvflare_cli" --connect-timeout "$NVFLARE_CLI_CONNECT_TIMEOUT" job logs "$job_id" \
  --sites all --tail 120 --startup-kit "$admin_kit" \
  2> >(tee "$run_dir/job-logs.stderr" >&2) | tee "$run_dir/job-logs.txt"; then
  echo "WARNING: completed-job log retrieval failed; inspect the Kubernetes participant logs" >&2
fi

((wait_rc == 0)) || die "hello-numpy did not complete successfully; see $run_dir"
job_status="$(jq -er 'select(.status == "ok") | .data.status | strings' <<<"$wait_json")" ||
  die "NVFlare wait did not return a successful terminal status"
[[ "$job_status" == FINISHED:COMPLETED* ]] ||
  die "hello-numpy ended with unexpected status $job_status"

report="$run_dir/report.txt"
cat >"$report" <<EOF
NVFLARE COCO HELLO-NUMPY JOB REPORT
==================================
Job ID: $job_id
Job: hello-numpy
Participants: $NVFLARE_SERVER_NAME, $NVFLARE_CLIENT_NAME
Launcher: ProcessJobLauncher (inside the existing Kata confidential VMs)
Server image: $server_image
Client image: $client_image
Status: $job_status
Submission: PASS
Terminal completion without NVFlare execution error: PASS
EOF
latest_run="$STATE_DIR/latest-nvflare-job-run"
[[ ! -e "$latest_run" || -L "$latest_run" ]] ||
  die "Refusing to replace non-symlink path: $latest_run"
ln -sfn "$run_dir" "$latest_run"

log "hello-numpy finished without error"
echo "Job ID: $job_id"
echo "Status: $job_status"
echo "Report: $report"
echo "Run artifacts: $run_dir"
