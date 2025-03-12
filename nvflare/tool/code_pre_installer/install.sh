# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash

# Default values
INSTALL_PREFIX="/opt/nvflare/jobs"
SHARE_LOCATION="/opt/nvflare/share"

# Print usage
usage() {
    echo "Usage: $0 --job-structure <path> --site-name <name> [--install-prefix <path>] [--share-location <path>]"
    echo
    echo "Arguments:"
    echo "  --job-structure   : (Required) Path to job structure zip file"
    echo "  --site-name      : (Required) Target site name (e.g., site-1, server)"
    echo "  --install-prefix : Installation directory for job code (default: ${INSTALL_PREFIX})"
    echo "  --share-location : Installation directory for shared resources (default: ${SHARE_LOCATION})"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-structure)
            JOB_STRUCTURE="$2"
            shift 2
            ;;
        --site-name)
            SITE_NAME="$2"
            shift 2
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --share-location)
            SHARE_LOCATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "${JOB_STRUCTURE}" ]] || [[ -z "${SITE_NAME}" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

# Execute Python installer
python3 -m nvflare.tool.code_pre_installer.install \
    --job-structure "${JOB_STRUCTURE}" \
    --site-name "${SITE_NAME}" \
    --install-prefix "${INSTALL_PREFIX}" \
    --share-location "${SHARE_LOCATION}"

exit $? 