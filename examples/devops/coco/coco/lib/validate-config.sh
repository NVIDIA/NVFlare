#!/usr/bin/env bash
# Source checkout entry point; packaging replaces this with the shared implementation.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/shared/validate-config.sh"
