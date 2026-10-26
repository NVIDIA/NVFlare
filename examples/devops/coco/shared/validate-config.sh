#!/usr/bin/env bash
# Validation helpers for operator-supplied build configuration.
# Configuration is trusted executable shell code, never an untrusted handoff.
config_error() { printf 'Configuration error: %s\n' "$*" >&2; exit 2; }
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

# Registry endpoints are independent of Trustee and may use any TLS port.
validate_registry_host() {
    local endpoint=$1 host port
    [[ $endpoint =~ ^[a-zA-Z0-9.-]+(:[0-9]+)?$ ]] || config_error 'Set REGISTRY_HOST to a hostname with an optional port, without a scheme or path.'
    host=${endpoint%%:*}
    validate_service_host "$host"
    if [[ $endpoint == *:* ]]; then
        port=${endpoint##*:}
        [[ ${#port} -le 5 ]] && ((10#$port >= 1 && 10#$port <= 65535)) || config_error 'Registry port must be between 1 and 65535.'
    fi
}
