#!/usr/bin/env bash
set -euo pipefail

# Launch a fresh Codex Auto-FL skill UX trial on the H100 host.
#
# The launcher prepares NVFlare and job dependencies before the agent starts.
# The agent should test the product skill behavior on an existing job.py, not
# rediscover Python, package, or skill-install setup details.

BASE=${AUTOFL_H100_BASE:-/scratch/hroth/Code/nvflare}
OUTPUT_BASE=${AUTOFL_H100_OUTPUT_BASE:-${BASE}/pr4780-autofl-output}
REPO_URL=${AUTOFL_H100_REPO_URL:-git@github.com:holgerroth/NVFlare.git}
BRANCH=${AUTOFL_H100_BRANCH:-codex/autofl-skill-v1}
JOB_PATH=${AUTOFL_H100_JOB:-examples/hello-world/hello-pt/job.py}
BOOTSTRAP_PYTHON=${AUTOFL_H100_BOOTSTRAP_PYTHON:-/scratch/hroth/Code/auto-fl/.venv312/bin/python}
SESSION_PREFIX=${AUTOFL_H100_SESSION_PREFIX:-pr4780-autofl-skill}
KILL_OLD=${AUTOFL_H100_KILL_OLD:-0}
SKIP_DEPS=${AUTOFL_H100_SKIP_DEPS:-0}
NVFLARE_EXTRA=${AUTOFL_H100_NVFLARE_EXTRA:-PT}
REQUIREMENTS=${AUTOFL_H100_REQUIREMENTS:-auto}
PARALLEL_CANDIDATES=${AUTOFL_H100_PARALLEL_CANDIDATES:-4}
CUDA_VISIBLE_DEVICES_VALUE=${AUTOFL_H100_CUDA_VISIBLE_DEVICES:-0}
OVERLAY_TGZ=${AUTOFL_H100_OVERLAY_TGZ:-}

TS=${AUTOFL_H100_TS:-$(date +%Y%m%d_%H%M%S)}
REPO=${AUTOFL_H100_REPO:-${BASE}/pr4780-autofl-skill-trial-${TS}}
OUT=${AUTOFL_H100_OUT:-${OUTPUT_BASE}/skill_trial_${TS}}
CODEX_TRIAL=${AUTOFL_H100_CODEX_HOME:-${OUT}/codex_home}
VENV_DIR=${AUTOFL_H100_VENV:-${OUT}/venv}
SESSION=${AUTOFL_H100_SESSION:-${SESSION_PREFIX}-${TS}}

find_bootstrap_python() {
  if [[ -x "${BOOTSTRAP_PYTHON}" ]]; then
    printf '%s\n' "${BOOTSTRAP_PYTHON}"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  echo "ERROR: no Python 3.12 bootstrap interpreter found." >&2
  echo "Set AUTOFL_H100_BOOTSTRAP_PYTHON to a Python 3.12 executable." >&2
  exit 2
}

resolve_path() {
  local base="$1"
  local path="$2"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${base}/${path}"
  fi
}

install_requirements_without_nvflare() {
  local requirements_file="$1"
  local filtered_requirements="${OUT}/requirements.filtered.txt"

  awk '
    /^[[:space:]]*($|#)/ { print; next }
    /^[[:space:]]*nvflare([[:space:]\[<=>!~].*)?$/ { next }
    { print }
  ' "${requirements_file}" > "${filtered_requirements}"

  if grep -Eq '^[[:space:]]*[^#[:space:]]' "${filtered_requirements}"; then
    "${PYTHON}" -m pip install -r "${filtered_requirements}"
  fi
}

if [[ "${KILL_OLD}" == "1" ]]; then
  tmux ls 2>/dev/null | awk -F: -v prefix="${SESSION_PREFIX}" '$1 ~ "^" prefix {print $1}' | while read -r old_session; do
    [[ -n "${old_session}" ]] && tmux kill-session -t "${old_session}" || true
  done
fi

mkdir -p "${BASE}" "${OUTPUT_BASE}" "${OUT}"

if [[ -e "${REPO}" ]]; then
  echo "ERROR: repo path already exists: ${REPO}" >&2
  exit 2
fi

git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO}"

if [[ -n "${OVERLAY_TGZ}" ]]; then
  if [[ ! -f "${OVERLAY_TGZ}" ]]; then
    echo "ERROR: overlay tarball not found: ${OVERLAY_TGZ}" >&2
    exit 2
  fi
  tar -xzf "${OVERLAY_TGZ}" -C "${REPO}"
fi

if [[ ! -d "${REPO}/skills/nvflare-autofl" ]]; then
  echo "ERROR: bundled skill not found: ${REPO}/skills/nvflare-autofl" >&2
  exit 2
fi

JOB_ABS=$(resolve_path "${REPO}" "${JOB_PATH}")
if [[ ! -f "${JOB_ABS}" ]]; then
  echo "ERROR: job.py not found: ${JOB_ABS}" >&2
  exit 2
fi

if [[ -n "${AUTOFL_H100_JOB_CWD:-}" ]]; then
  JOB_CWD=$(resolve_path "${REPO}" "${AUTOFL_H100_JOB_CWD}")
else
  JOB_CWD=$(dirname "${JOB_ABS}")
fi
if [[ ! -d "${JOB_CWD}" ]]; then
  echo "ERROR: job cwd not found: ${JOB_CWD}" >&2
  exit 2
fi

PY_BOOTSTRAP=$(find_bootstrap_python)
PYTHON="${VENV_DIR}/bin/python"

if [[ "${SKIP_DEPS}" != "1" ]]; then
  "${PY_BOOTSTRAP}" -m venv "${VENV_DIR}"
  "${PYTHON}" -m pip install --upgrade pip

  if [[ "${REQUIREMENTS}" == "auto" ]]; then
    REQUIREMENTS="${JOB_CWD}/requirements.txt"
  elif [[ "${REQUIREMENTS}" != "none" ]]; then
    REQUIREMENTS=$(resolve_path "${REPO}" "${REQUIREMENTS}")
  fi

  if [[ "${REQUIREMENTS}" != "none" && -f "${REQUIREMENTS}" ]]; then
    install_requirements_without_nvflare "${REQUIREMENTS}"
  fi

  "${PYTHON}" -m pip uninstall -y nvflare-nightly >/dev/null 2>&1 || true
  "${PYTHON}" -m pip uninstall -y nvflare >/dev/null 2>&1 || true
  if [[ -n "${NVFLARE_EXTRA}" ]]; then
    "${PYTHON}" -m pip install -e "${REPO}[${NVFLARE_EXTRA}]"
  else
    "${PYTHON}" -m pip install -e "${REPO}"
  fi
fi

"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)'
"${PYTHON}" -c 'import msgpack, nvflare; print("nvflare", nvflare.__file__)'

mkdir -p "${CODEX_TRIAL}/skills"
cp "${HOME}/.codex/config.toml" "${CODEX_TRIAL}/config.toml"
if [[ -f "${HOME}/.codex/auth.json" ]]; then
  ln -sf "${HOME}/.codex/auth.json" "${CODEX_TRIAL}/auth.json"
fi
chmod 700 "${CODEX_TRIAL}" 2>/dev/null || true

cp -R "${REPO}/skills/nvflare-autofl" "${CODEX_TRIAL}/skills/nvflare-autofl"

cat >> "${CODEX_TRIAL}/config.toml" <<EOF

[projects."${REPO}"]
trust_level = "trusted"

[projects."${JOB_CWD}"]
trust_level = "trusted"
EOF

PROMPT=${AUTOFL_H100_PROMPT:-"Optimize ./$(basename "${JOB_ABS}") for accuracy in sim."}

cat > "${OUT}/initial_prompt.txt" <<EOF
Select: NVFlare Auto-FL skill

Prompt: ${PROMPT}
EOF

cat > "${OUT}/session.env" <<EOF
SESSION=${SESSION}
REPO=${REPO}
JOB=${JOB_ABS}
JOB_CWD=${JOB_CWD}
OUT=${OUT}
CODEX_TRIAL=${CODEX_TRIAL}
VENV_DIR=${VENV_DIR}
PYTHON=${PYTHON}
BRANCH=${BRANCH}
PARALLEL_CANDIDATES=${PARALLEL_CANDIDATES}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}
EOF

tmux new-session -d -s "${SESSION}" -c "${JOB_CWD}" \
  "export VIRTUAL_ENV='${VENV_DIR}'; \
   export PATH='${VENV_DIR}/bin':\"\$PATH\"; \
   export PYTHON='${PYTHON}'; \
   export PARALLEL_CANDIDATES='${PARALLEL_CANDIDATES}'; \
   export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_VALUE}'; \
   CODEX_HOME='${CODEX_TRIAL}' codex -a on-request \
     -c 'approvals_reviewer=\"auto_review\"' \
     -c 'shell_environment_policy.inherit=\"all\"' \
     -s workspace-write \
     --add-dir '${REPO}' \
     --add-dir '${JOB_CWD}' \
     --add-dir '${OUT}' \
     --add-dir /tmp \
     --no-alt-screen \
     -C '${JOB_CWD}' \"\$(cat '${OUT}/initial_prompt.txt')\""

tmux pipe-pane -o -t "${SESSION}" "cat >> '${OUT}/codex-tui.log'"

cat <<EOF
Session: ${SESSION}
Repo: ${REPO}
Job: ${JOB_ABS}
Job cwd: ${JOB_CWD}
Output: ${OUT}
Python: ${PYTHON}
Prompt: ${PROMPT}
Attach with: tmux attach -t ${SESSION}
Monitor with: source ${OUT}/session.env && tmux capture-pane -pt "\$SESSION" -S -120
EOF
