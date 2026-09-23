#!/usr/bin/env bash
# Source-only local preflight; no cluster access or privileged operations.
rehearsal_preflight() {
    preflight_errors=()
    preflight_error() { preflight_errors+=("$*"); }
    finish_preflight() {
        if (( ${#preflight_errors[@]} )); then
            printf 'Stage 07 preflight found %s problem(s):\n' "${#preflight_errors[@]}" >&2
            printf '  - %s\n' "${preflight_errors[@]}" >&2
            printf 'No rehearsal started. Preserve existing evidence; use a new PLATFORM_PROFILE for a fresh run.\n' >&2
            exit 1
        fi
    }
    for command in awk base64 chmod cmp curl cut dirname docker find grep gzip head install ip kubectl openssl \
        python3 realpath sed sha256sum skopeo sleep sort tar xargs; do
        command -v "$command" >/dev/null 2>&1 || preflight_error "missing command: $command"
    done
    # Do not invoke sudo or contact the cluster until local preflight has passed.
    command -v ctr >/dev/null 2>&1 || [[ -x /usr/bin/ctr ]] || preflight_error 'missing command: ctr'
    command -v crictl >/dev/null 2>&1 || preflight_error 'missing command: crictl (install cri-tools)'
    for config in "${BASE_CONFIG}" "${APPROVAL_ENV}"; do
        [[ -f "$config" && -r "$config" && -s "$config" ]] || preflight_error "missing, empty or unreadable configuration: $config"
    done
    # Dependent checks require readable trusted inputs. Report all problems known
    # so far when these cannot be loaded; never guess a profile path.
    if [[ ! -f "$BASE_CONFIG" || ! -r "$BASE_CONFIG" || ! -s "$BASE_CONFIG" ||
          ! -f "$APPROVAL_ENV" || ! -r "$APPROVAL_ENV" || ! -s "$APPROVAL_ENV" ]]; then
        finish_preflight
    fi

    # shellcheck source=/dev/null
    source "${BASE_CONFIG}"
    [[ "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || preflight_error 'invalid PLATFORM_PROFILE'
    [[ "${RUNTIME_CLASS-}" == 'kata-qemu-nvidia-gpu-snp' ]] || preflight_error 'unexpected RUNTIME_CLASS'
    [[ "${PLATFORM_WORK_ROOT:-}" == /* ]] || preflight_error 'PLATFORM_WORK_ROOT must be an absolute path'
    if [[ ! "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ || "${PLATFORM_WORK_ROOT:-}" != /* ]]; then
        finish_preflight
    fi
    PROFILE_DIR="${PLATFORM_WORK_ROOT}/${PLATFORM_PROFILE}"
    if [[ "$(dirname -- "${APPROVAL_ENV}")" != "${PROFILE_DIR}" ]]; then
        preflight_error 'APPROVAL-ENV must be inside the selected platform profile'
        finish_preflight
    fi

    COLLECTOR_DIR="${PROFILE_DIR}/rehearsal-collector-build"
    REHEARSAL_EVIDENCE="${PROFILE_DIR}/rehearsal-evidence.txt"
    for artifact in "$COLLECTOR_DIR" "$PROFILE_DIR/reported-tcb-evidence" "$REHEARSAL_EVIDENCE"; do
        [[ ! -e "$artifact" && ! -L "$artifact" ]] || preflight_error "refusing to overwrite: $artifact"
    done
    if [[ -z "${REHEARSAL_WORKLOAD_YAML:-}" ]]; then
        preflight_error 'REHEARSAL_WORKLOAD_YAML is required for the approved workload rehearsal (stages 05-10)'
    else
        [[ -f "$REHEARSAL_WORKLOAD_YAML" && -r "$REHEARSAL_WORKLOAD_YAML" && -s "$REHEARSAL_WORKLOAD_YAML" ]] \
            || preflight_error "missing, empty or unreadable source Pod YAML: $REHEARSAL_WORKLOAD_YAML"
    fi
    for input in "$PROFILE_DIR/approved-launch-profile.json" \
        "$SCRIPT_DIR/rehearsal-collector/Dockerfile" "$SCRIPT_DIR/rehearsal-collector/collect-snp-evidence.sh" \
        "$SCRIPT_DIR/capture-running-launch.py" "$SCRIPT_DIR/record-reported-tcb.sh"; do
        [[ -f "$input" && -r "$input" && -s "$input" ]] || preflight_error "missing, empty or unreadable input: $input"
    done
    # Read only the reviewed baseline in a subshell: approval variables must not
    # override the base configuration used by the collector. Do not print values.
    if approval_errors="$(
        source "$APPROVAL_ENV" || exit 1
        for field in BOOTLOADER TEE SNP MICROCODE; do
            variable="SNP_MIN_REPORTED_TCB_${field}"
            value="${!variable-}"
            if ! [[ "$value" =~ ^(0|[1-9][0-9]{0,2})$ ]] || ((10#$value > 255)); then
                printf 'set independently approved decimal uint8 %s before stage 07\n' "$variable"
            fi
        done
    )"; then
        while IFS= read -r problem; do
            [[ -z "$problem" ]] || preflight_error "$problem"
        done <<< "$approval_errors"
    else
        preflight_error 'could not load the trusted approval configuration'
    fi
    finish_preflight
}
