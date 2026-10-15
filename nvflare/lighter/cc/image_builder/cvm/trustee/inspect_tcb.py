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

"""Inspect candidate TDX TCB fields; this command never approves or imports them."""

from ..common.evidence import tdx_tcb


def inspect_tcb(evidence):
    candidate = tdx_tcb(evidence)
    return {
        "unapproved_candidate_tcb": {name: [value] for name, value in candidate.items()},
        "review_required": "Validate platform endorsements and TCB status; obtain advisory IDs from a real signed-quote appraisal. This output is not an approved reference file.",
    }
