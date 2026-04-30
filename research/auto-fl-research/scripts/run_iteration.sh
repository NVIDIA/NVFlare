#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
RESULTS_TSV=${RESULTS_TSV:-results.tsv}
RUN_LOG=${RUN_LOG:-run.log}
RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-600}
RUN_ITERATION_LOG_RESULTS=${RUN_ITERATION_LOG_RESULTS:-1}
RUN_ITERATION_REQUIRE_SCORE=${RUN_ITERATION_REQUIRE_SCORE:-1}
DESCRIPTION=""
TARGET=""
EXPECTED_RESULTS_HEADER=$'commit\tscore\truntime_seconds\tbudget\tstatus\ttarget\tdescription\tartifacts'
OLD_RESULTS_HEADER=$'commit\tscore\tbudget\tstatus\ttarget\tdescription\tartifacts'

ensure_results_tsv() {
  if [[ ! -f "${RESULTS_TSV}" ]]; then
    cp templates/results_header.tsv "${RESULTS_TSV}"
    return
  fi

  local current_header
  current_header=$(head -n 1 "${RESULTS_TSV}")
  if [[ "${current_header}" == "${EXPECTED_RESULTS_HEADER}" ]]; then
    return
  fi

  if [[ "${current_header}" == "${OLD_RESULTS_HEADER}" ]]; then
    local tmp_results
    tmp_results=$(mktemp "${RESULTS_TSV}.XXXXXX")
    {
      printf '%s\n' "${EXPECTED_RESULTS_HEADER}"
      tail -n +2 "${RESULTS_TSV}" | awk -F '\t' 'BEGIN { OFS = "\t" } { print $1, $2, "", $3, $4, $5, $6, $7 }'
    } > "${tmp_results}"
    mv "${tmp_results}" "${RESULTS_TSV}"
    return
  fi

  echo "Unknown ${RESULTS_TSV} header: ${current_header}" >&2
  exit 2
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

"${PYTHON}" scripts/validate_contract.py client.py

COMMAND=("${PYTHON}" job.py --cross_site_eval "$@")
printf 'Running: %q ' "${COMMAND[@]}"
printf '\n'
echo "log=${RUN_LOG}"
mkdir -p "$(dirname "${RUN_LOG}")"

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
RESULT_DIR=$(grep '^Results:' "${RUN_LOG}" | tail -n 1 | sed 's/^Results:[[:space:]]*//' || true)

if [[ $RC -ne 0 || -z "${RESULT_DIR}" ]]; then
  echo "Run failed. See ${RUN_LOG}"
  tail -n 50 "${RUN_LOG}" || true
  if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
    ensure_results_tsv
    printf '%s\t0.000000\t%s\t%s\tcrash\t%s\t%s\t%s\n' "$COMMIT" "$RUNTIME_SECONDS" "$BUDGET" "$TARGET" "$DESCRIPTION" "${RUN_LOG}" >> "${RESULTS_TSV}"
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
    ensure_results_tsv
    printf '%s\t0.000000\t%s\t%s\tcrash\t%s\t%s\t%s\n' "$COMMIT" "$RUNTIME_SECONDS" "$BUDGET" "$TARGET" "$DESCRIPTION" "$RESULT_DIR" >> "${RESULTS_TSV}"
  fi
  echo "${SCORE_OUTPUT}" >&2
  exit "${SCORE_RC}"
fi

SCORE=${SCORE_OUTPUT}
echo "score=${SCORE}"
echo "result_dir=${RESULT_DIR}"

if [[ "${RUN_ITERATION_LOG_RESULTS}" != "0" ]]; then
  ensure_results_tsv
  printf '%s\t%s\t%s\t%s\tcandidate\t%s\t%s\t%s\n' "$COMMIT" "$SCORE" "$RUNTIME_SECONDS" "$BUDGET" "$TARGET" "$DESCRIPTION" "$RESULT_DIR" >> "${RESULTS_TSV}"
  echo "Appended candidate result to ${RESULTS_TSV}"
else
  echo "Skipped ${RESULTS_TSV} logging because RUN_ITERATION_LOG_RESULTS=0"
fi
