#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${BENCHMARK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [COMMAND] --prompt PATH [--training-code PATH] [--results-root PATH] [PATH]

Commands:
  pair             Run paired skills/no-skills benchmark cases. This is the default when COMMAND is omitted.
  one              Run one benchmark case. Use --mode with_skills or without_skills.
  scenario         Run a compiled scenario YAML.
  replay           Rebuild parser artifacts and scenario reports from captured results.
  interactive      Start an interactive benchmark container.
  with-skills      Shortcut that runs one benchmark case with MODE=with_skills.
  without-skills   Shortcut that runs one benchmark case with MODE=without_skills.

Examples:
  ./bin/run.sh --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --training-code /path/to/job-folder
  ./bin/run.sh pair --agent claude --model MODEL --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --agent-home /path/to/agent-home --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --no-agent-auth-mount --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --results-root /path/to/results /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --output-dir /path/to/exact-run-dir /path/to/job-folder
  ./bin/run.sh scenario /path/to/scenario.yaml --output-dir /path/to/exact-run-dir
  ./bin/run.sh replay /path/to/existing-run-dir
  ./bin/run.sh one --mode with_skills --prompt /path/to/prompt.txt /path/to/job-folder
EOF
}

# On macOS, Claude Code stores its API key in the Keychain rather than a file.
# Extract it so passthrough_env can forward it into the benchmark container.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && command -v security &>/dev/null; then
  _keychain_key="$(security find-generic-password -s "Claude Code" -w 2>/dev/null || true)"
  if [[ -n "${_keychain_key}" ]]; then
    export ANTHROPIC_API_KEY="${_keychain_key}"
  fi
  unset _keychain_key
fi

# On Linux, an API-key login leaves ~/.claude/.credentials.json empty and keeps
# the key in ~/.claude.json (primaryApiKey); the keyring is not used. Extract it
# so passthrough_env can forward it into the container, mirroring the macOS path.
# Uses only the Python standard library, so the benchmark needs no extra deps.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  _claude_json="${HOME}/.claude.json"
  if [[ -f "${_claude_json}" ]]; then
    _api_key="$(python3 -c 'import json,sys
try:
    key = json.load(open(sys.argv[1])).get("primaryApiKey") or ""
except Exception:
    key = ""
print(key if key.startswith("sk-ant-") else "")' "${_claude_json}" 2>/dev/null || true)"
    if [[ -n "${_api_key}" ]]; then
      export ANTHROPIC_API_KEY="${_api_key}"
    fi
    unset _api_key
  fi
  unset _claude_json
fi

# A single API key is a complete credential. If we have one, drop any OAuth /
# auth-token vars so a stale value cannot be forwarded into the container and
# override the key (symptom: "401 Invalid bearer token", apiKeySource=none).
# This makes the run immune to a stale token cached in the launching shell.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN
fi

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
  command="$1"
  shift
elif [[ "$1" == -* ]]; then
  command="pair"
else
  command="$1"
  shift
fi

case "${command}" in
  one|run-one|single)
    exec python3 -m skills.harness.host.runner run-one "$@"
    ;;
  pair)
    exec python3 -m skills.harness.host.runner pair "$@"
    ;;
  scenario)
    exec python3 -m skills.harness.host.runner scenario "$@"
    ;;
  replay)
    exec python3 -m skills.harness.host.runner replay "$@"
    ;;
  interactive|shell)
    exec python3 -m skills.harness.host.runner interactive "$@"
    ;;
  with-skills)
    MODE=with_skills USE_PREINSTALLED_SKILLS=true exec python3 -m skills.harness.host.runner run-one "$@"
    ;;
  without-skills)
    MODE=without_skills USE_PREINSTALLED_SKILLS=false exec python3 -m skills.harness.host.runner run-one "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
