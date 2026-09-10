#!/usr/bin/env bash
# Configuration is trusted executable shell code, never an untrusted handoff.
config_error() { printf 'Configuration error: %s\n' "$*" >&2; exit 2; }
validate_target_host() {
    [[ -n ${EXPECTED_HOSTNAME:-} ]] || config_error 'Set EXPECTED_HOSTNAME to the intended machine hostname -f.'
    [[ ${EXPECTED_HOSTNAME} == "$(hostname -f)" ]] || config_error 'EXPECTED_HOSTNAME does not match this machine.'
}
validate_service_host() {
    local host=$1 label
    [[ ${#host} -le 253 && $host =~ ^[a-zA-Z0-9.-]+$ && $host == *.* && $host != *..* ]] || config_error 'Set a valid secure-services DNS name.'
    case ${host,,} in
        example.com|*.example.com|example.net|*.example.net|example.org|*.example.org|*.invalid|*.test)
            config_error 'Replace the documentation endpoint with an authenticated deployment DNS name.' ;;
    esac
    local -a labels
    IFS=. read -r -a labels <<< "$host"
    for label in "${labels[@]}"; do
        [[ ${#label} -le 63 && $label =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$ ]] || config_error 'Invalid DNS label.'
    done
    [[ $host =~ [a-zA-Z] && $host != *. ]] || config_error 'Use a DNS name without a trailing dot, not an IP address.'
}
validate_private_root() {
    local directory=$1
    [[ $directory == /* && $directory != / && $directory != "$HOME" && $directory != /home && $directory != /tmp ]] || config_error 'Use a dedicated absolute state directory.'
    [[ $directory != *'/../'* && $directory != */.. && $directory != *'/./'* && $directory != */. ]] || config_error 'State paths must not contain dot components.'
}
