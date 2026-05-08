#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
RESULTS_TSV=${RESULTS_TSV:-results.tsv}
RUN_LOG=${RUN_LOG:-run.log}
RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-1200}
RUN_ITERATION_LOG_RESULTS=${RUN_ITERATION_LOG_RESULTS:-1}
RUN_ITERATION_REQUIRE_SCORE=${RUN_ITERATION_REQUIRE_SCORE:-1}
JOB_SCRIPT=${JOB_SCRIPT:-job.py}
CLIENT_CONTRACT_PATH=${CLIENT_CONTRACT_PATH:-client.py}
DESCRIPTION=""
TARGET=""

append_result() {
  local score="$1"
  local status="$2"
  local artifacts="$3"
  "${PYTHON}" scripts/append_result.py \
    --results="${RESULTS_TSV}" \
    --commit="${COMMIT}" \
    --score="${score}" \
    --runtime-seconds="${RUNTIME_SECONDS}" \
    --budget="${BUDGET}" \
    --status="${status}" \
    --target="${TARGET}" \
    --description="${DESCRIPTION}" \
    --artifacts="${artifacts}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --description)
      DESCRIPTION="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --no-log-results)
      RUN_ITERATION_LOG_RESULTS=0
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DESCRIPTION}" ]]; then
  echo "--description is required" >&2
  exit 2
fi
if [[ -z "${TARGET}" ]]; then
  echo "--target is required" >&2
  exit 2
fi

if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: run_iteration.sh with result logging must run inside a git clone" >&2
    exit 2
  fi

  CURRENT_BRANCH=$(git branch --show-current)
  if [[ "${CURRENT_BRANCH}" != autoresearch/* ]]; then
    echo "ERROR: current branch is '${CURRENT_BRANCH}', not autoresearch/<tag>." >&2
    echo "Run 'bash scripts/init_run.sh <tag>' before launching campaign experiments." >&2
    echo "Use --no-log-results only for smoke checks that should not write results.tsv." >&2
    exit 2
  fi

  "${PYTHON}" scripts/append_result.py --results="${RESULTS_TSV}" --init-only
fi

"${PYTHON}" scripts/validate_contract.py "${CLIENT_CONTRACT_PATH}"

COMMAND=("${PYTHON}" "${JOB_SCRIPT}" --cross_site_eval "$@")
printf 'Running: %q ' "${COMMAND[@]}"
printf '\n'
echo "log=${RUN_LOG}"
mkdir -p "$(dirname "${RUN_LOG}")"

CLEANUP_RESULT_DIR_FILE=0
if [[ -z "${AUTOFL_RESULT_DIR_FILE:-}" ]]; then
  AUTOFL_RESULT_DIR_FILE=$(mktemp "${RUN_LOG}.result_dir.XXXXXX")
  CLEANUP_RESULT_DIR_FILE=1
fi
export AUTOFL_RESULT_DIR_FILE
rm -f "${AUTOFL_RESULT_DIR_FILE}"
trap 'if [[ "${CLEANUP_RESULT_DIR_FILE}" == "1" ]]; then rm -f "${AUTOFL_RESULT_DIR_FILE}"; fi' EXIT

START_SECONDS=$(date +%s)
set +e
if [[ "${RUN_TIMEOUT_SECONDS}" != "0" ]]; then
  "${PYTHON}" scripts/run_with_timeout.py --timeout "${RUN_TIMEOUT_SECONDS}" --log "${RUN_LOG}" -- "${COMMAND[@]}"
else
  "${COMMAND[@]}" > "${RUN_LOG}" 2>&1
fi
RC=$?
set -e
END_SECONDS=$(date +%s)
RUNTIME_SECONDS=$((END_SECONDS - START_SECONDS))

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo no-git)
BUDGET=$(printf '%s ' "$@" | sed 's/[[:space:]]\+$//')
RESULT_DIR=$(sed -n '1p' "${AUTOFL_RESULT_DIR_FILE}" 2>/dev/null || true)

if [[ $RC -ne 0 || -z "${RESULT_DIR}" ]]; then
  echo "Run failed. See ${RUN_LOG}"
  tail -n 50 "${RUN_LOG}" || true
  if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
    append_result "0.000000" "crash" "${RUN_LOG}"
  else
    echo "Skipped ${RESULTS_TSV} logging because RUN_ITERATION_LOG_RESULTS=0"
  fi
  exit 1
fi

set +e
SCORE_OUTPUT=$("${PYTHON}" scripts/extract_score.py "$RESULT_DIR" 2>&1)
SCORE_RC=$?
set -e
if [[ ${SCORE_RC} -ne 0 ]]; then
  echo "Score extraction failed for result_dir=${RESULT_DIR}"
  if [[ "${RUN_ITERATION_REQUIRE_SCORE}" == "0" ]]; then
    SCORE_WARNING=$(printf '%s' "${SCORE_OUTPUT}" | tail -n 1)
    echo "score_warning=${SCORE_WARNING}"
    echo "Continuing because RUN_ITERATION_REQUIRE_SCORE=0"
    echo "result_dir=${RESULT_DIR}"
    if [[ "${RUN_ITERATION_LOG_RESULTS}" == "0" ]]; then
      echo "Skipped ${RESULTS_TSV} logging because RUN_ITERATION_LOG_RESULTS=0"
    fi
    exit 0
  fi
  if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
    append_result "0.000000" "crash" "$RESULT_DIR"
  fi
  echo "${SCORE_OUTPUT}" >&2
  exit "${SCORE_RC}"
fi

SCORE=${SCORE_OUTPUT}
echo "score=${SCORE}"
echo "result_dir=${RESULT_DIR}"

if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
  append_result "$SCORE" "candidate" "$RESULT_DIR"
  echo "Appended candidate result to ${RESULTS_TSV}"
else
  echo "Skipped ${RESULTS_TSV} logging because RUN_ITERATION_LOG_RESULTS=0"
fi
