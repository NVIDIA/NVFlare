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

"""Connection settings shared by hosted and self-hosted chat endpoints."""

import os

from openai import OpenAI


def create_llm_client() -> OpenAI:
    base_url = (
        os.environ.get("MEMCO_LLM_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = os.environ.get("MEMCO_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "MEMCO_LLM_API_KEY is required for live or record mode. "
            "For a local endpoint without authentication, set MEMCO_LLM_API_KEY=unused."
        )
    return OpenAI(base_url=base_url, api_key=api_key)
