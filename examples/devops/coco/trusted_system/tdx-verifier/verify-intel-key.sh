#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
umask 077
[[ $# == 2 ]] || { printf 'Usage: %s INPUT_ASC OUTPUT_GPG\n' "$0" >&2; exit 2; }
[[ -f $1 && -s $1 && ! -e $2 ]] || { printf 'Invalid key input/output\n' >&2; exit 1; }
KEY_HOME="$(mktemp -d /tmp/coco-intel-key.XXXXXX)"
FINGERPRINT='150434D1488BF80308B69398E5C7F0FA1C6C6C3C'
# No ambient gpg.conf, keyring, ownertrust or configured keyserver is consulted.
# Inspect every primary key, not just the first fingerprint in a key bundle.
gpg --no-options --homedir "${KEY_HOME}" --batch --show-keys --with-colons --with-fingerprint "$1" |
    awk -F: -v expected="${FINGERPRINT}" -v now="$(date +%s)" '
        $1 == "sec" || $1 == "ssb" { bad=1 }
        $1 == "pub" {
            count++; primary=1
            if ($2 ~ /[erid]/ || ($7 != "" && $7 != "0" && $7 <= now)) bad=1
        }
        $1 == "fpr" && primary {
            if ($10 != expected) bad=1
            fingerprints++; primary=0
        }
        END { exit !(count == 1 && fingerprints == 1 && !bad) }
    ' || { printf 'Intel package signing key is not the single approved unexpired public key\n' >&2; exit 1; }
gpg --no-options --homedir "${KEY_HOME}" --batch --dearmor --output "$2" "$1"
chmod 0644 -- "$2"
