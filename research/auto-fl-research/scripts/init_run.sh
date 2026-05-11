#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-}
if [[ -z "${TAG}" ]]; then
  TAG="h100-baseline-$(date +%Y%m%d)"
fi
TAG=$(printf '%s' "${TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//')

if [[ -z "${TAG}" ]]; then
  echo "ERROR: empty run tag after normalization" >&2
  exit 2
fi

if [[ ! "${TAG}" =~ [0-9]{8} ]]; then
  echo "ERROR: run tag must include a YYYYMMDD date, for example h100-fedavgm-$(date +%Y%m%d)" >&2
  exit 2
fi

TAG_WITHOUT_DATE=$(printf '%s' "${TAG}" | sed -E 's/(^|-)[0-9]{8}($|-)/-/g; s/[-_.]//g')
if [[ ${#TAG_WITHOUT_DATE} -lt 6 ]]; then
  echo "ERROR: run tag must include a descriptive campaign topic, for example h100-fedavgm-$(date +%Y%m%d) or h100-archsearch-$(date +%Y%m%d)" >&2
  exit 2
fi

BRANCH="autoresearch/${TAG}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: init_run.sh must run inside a git clone so experiments have branch and commit provenance" >&2
  exit 2
fi

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  current_branch=$(git branch --show-current)
  if [[ "${current_branch}" == "${BRANCH}" ]]; then
    echo "Branch ${BRANCH} already checked out"
  else
    git checkout "${BRANCH}"
    echo "Switched to existing branch ${BRANCH}"
  fi
else
  git checkout -b "${BRANCH}"
  echo "Created branch ${BRANCH}"
fi

current_branch=$(git branch --show-current)
if [[ "${current_branch}" != "${BRANCH}" ]]; then
  echo "ERROR: expected to be on ${BRANCH}, but current branch is ${current_branch}" >&2
  exit 2
fi

if [[ ! -f results.tsv ]]; then
  cp templates/results_header.tsv results.tsv
  echo "Initialized results.tsv"
else
  echo "results.tsv already exists; leaving it unchanged"
fi

echo "Run initialized. Next steps:"
echo "  make validate"
echo "  make smoke"
echo "  bash scripts/run_iteration.sh --description \"baseline\" --target client.py -- <budget args>"
