#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || {
    printf 'Usage: %s RECEIVED_CREDENTIAL_DIRECTORY\n' "$0" >&2
    exit 2
}

SOURCE_DIR="$(realpath -e -- "$1")"
[[ -d "${SOURCE_DIR}" && ! -L "$1" ]] || {
    printf 'Credential source must be a real directory, not a symlink.\n' >&2
    exit 1
}

mapfile -t ENTRIES < <(find "${SOURCE_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
[[ "${#ENTRIES[@]}" -eq 2 && "${ENTRIES[0]}" == password && "${ENTRIES[1]}" == username ]] || {
    printf 'Credential directory must contain exactly username and password.\n' >&2
    exit 1
}

for name in username password; do
    [[ -f "${SOURCE_DIR}/${name}" && ! -L "${SOURCE_DIR}/${name}" && -s "${SOURCE_DIR}/${name}" ]] || {
        printf '%s must be a nonempty regular file, not a symlink.\n' "${name}" >&2
        exit 1
    }
done

[[ "$(tr -d '\r\n' < "${SOURCE_DIR}/username")" == coco-publisher ]] || {
    printf 'Unexpected registry publisher username.\n' >&2
    exit 1
}

DESTINATION="${HOME}/coco-workload-owner/secrets/registry"
[[ ! -e "${DESTINATION}/username" && ! -e "${DESTINATION}/password" ]] || {
    printf 'Publisher credential is already installed; refusing to overwrite it.\n' >&2
    exit 1
}

install -d -m 0700 "${DESTINATION}"
install -m 0600 "${SOURCE_DIR}/username" "${DESTINATION}/username"
install -m 0600 "${SOURCE_DIR}/password" "${DESTINATION}/password"

printf 'Installed registry publisher credential in %s.\n' "${DESTINATION}"
printf 'The password was not displayed. Remove the received copy securely.\n'
