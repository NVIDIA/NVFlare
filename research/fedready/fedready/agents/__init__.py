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

"""FedReady agents.

The package is split by concern:

* ``roles``: ``ServerAgent`` and ``ClientAgent``, which compose prompts, dispatch them to
  the Codex backend and validate agent output.
* ``guardrail``: allow-list rules, ``GuardrailAgent`` and redaction helpers.
* ``visual_qc``: visual QC templates, local VLM review and sample voting.
* ``local_vlm``: transport for the client-local vision-language model.
* ``bridge``, ``local_adapter``, ``preflight`` and ``training_reference``: the Codex
  backend, client-local adapter pipeline, runtime preflight and NVFlare example selection.
"""

from fedready.agents.guardrail import ACTION_CHECKS, ALLOW_LIST_RULES, GuardrailAgent, GuardrailDecision
from fedready.agents.local_vlm import DEFAULT_LOCAL_VISION_MODEL, LOCAL_VISION_API_BASE_URL
from fedready.agents.roles import AgentTurn, ClientAgent, ServerAgent

__all__ = [
    "ACTION_CHECKS",
    "ALLOW_LIST_RULES",
    "DEFAULT_LOCAL_VISION_MODEL",
    "LOCAL_VISION_API_BASE_URL",
    "AgentTurn",
    "ClientAgent",
    "GuardrailAgent",
    "GuardrailDecision",
    "ServerAgent",
]
