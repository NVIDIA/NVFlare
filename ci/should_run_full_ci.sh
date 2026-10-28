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
# pre-merge jobs must run. Required GitHub checks still start for docs-only
# changes, but their expensive steps use this result to exit quickly.
set -euo pipefail

saw_path=false
while IFS= read -r changed_path || [ -n "$changed_path" ]; do
    saw_path=true
    case "$changed_path" in
        # Skill instructions and test inputs (including Markdown fixtures) are
        # not documentation-only: their consumers need full validation.
        skills/* | tests/*)
            echo true
            exit 0
            ;;
        # Docs read by unit tests; keep this list in sync with their inputs.
        integration/nemo/examples/peft/README.md | \
            docs/user_guide/data_scientist_guide/available_recipes.rst | \
            examples/hello-world/agent-skills/pytorch-conversion/README.md | \
            examples/hello-world/agent-skills/lightning-conversion/README.md | \
            examples/hello-world/agent-skills/huggingface-conversion/README.md)
            echo true
            exit 0
            ;;
        *.md | *.rst)
            ;;
        docs/*.css | docs/*.gif | docs/*.html | docs/*.ico | docs/*.in | \
            docs/*.jpeg | docs/*.jpg | docs/*.json | docs/*.png | docs/*.svg | \
            docs/*.txt | docs/*.webp | docs/Makefile)
            ;;
        CODE_OF_CONDUCT.md | CONTRIBUTING.md | CITATION.cff | LICENSE)
            ;;
        *)
            echo true
            exit 0
            ;;
    esac
done

# Fail closed if the event did not provide any paths.
if [ "$saw_path" = false ]; then
    echo true
else
    echo false
fi
