#!/usr/bin/env bash
# Shared shell primitives; no service or host setup.
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"; }
need_file() { [[ -s "$1" ]] || die "required file is missing or empty: $1"; }
