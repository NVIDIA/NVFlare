#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Read changed repository paths from stdin and print whether the expensive
# Tier 1 skill-security unit tests must run. Keep the policy here so it can be
# covered by unit tests instead of existing only as an inline workflow command.
set -euo pipefail

while IFS= read -r changed_path || [ -n "$changed_path" ]; do
    case "$changed_path" in
        skills/* | tests/unit_test/tool/agent_skill_checks/* | dev_tools/agent/skills/* | .github/workflows/premerge.yml | ci/should_run_skill_tier1.sh | setup.cfg)
            echo true
            exit 0
            ;;
    esac
done

echo false
