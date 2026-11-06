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

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from mas.llm_config import create_llm_client  # noqa: E402


class LLMConfigTest(unittest.TestCase):
    def test_default_endpoint_is_openai(self):
        with patch.dict(os.environ, {"MEMCO_LLM_API_KEY": "cloud-test-key"}, clear=True):
            with create_llm_client() as client:
                self.assertEqual("https://api.openai.com/v1/", str(client.base_url))

    def test_memco_variables_override_legacy_variables(self):
        environment = {
            "MEMCO_LLM_BASE_URL": "http://gpu-host:8000/v1",
            "MEMCO_LLM_API_KEY": "unused",
            "OPENAI_API_BASE": "https://legacy.example/v1",
            "OPENAI_BASE_URL": "https://other.example/v1",
            "OPENAI_API_KEY": "legacy-test-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            with create_llm_client() as client:
                self.assertEqual("http://gpu-host:8000/v1/", str(client.base_url))
                self.assertEqual("unused", client.api_key)

    def test_legacy_endpoint_variables_remain_supported(self):
        for variable in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
            with self.subTest(variable=variable):
                with patch.dict(
                    os.environ, {variable: "http://gpu-host:8000/v1", "OPENAI_API_KEY": "legacy-test-key"}, clear=True
                ):
                    with create_llm_client() as client:
                        self.assertEqual("http://gpu-host:8000/v1/", str(client.base_url))
                        self.assertEqual("legacy-test-key", client.api_key)

    def test_missing_key_explains_local_server_placeholder(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MEMCO_LLM_API_KEY=unused"):
                create_llm_client()

    def test_settings_are_read_when_client_is_created(self):
        with patch.dict(os.environ, {"MEMCO_LLM_API_KEY": "test-key"}, clear=True):
            with create_llm_client() as cloud:
                os.environ["MEMCO_LLM_BASE_URL"] = "http://gpu-host:8000/v1"
                with create_llm_client() as local:
                    self.assertEqual("https://api.openai.com/v1/", str(cloud.base_url))
                    self.assertEqual("http://gpu-host:8000/v1/", str(local.base_url))

    def test_chat_requests_use_selected_endpoint_key_and_model(self):
        cases = [
            ("https://api.openai.com/v1", "cloud-test-key", "gpt-4o-mini"),
            ("http://gpu-host:8000/v1", "unused", "qwen32b-api"),
        ]
        for base_url, api_key, model in cases:
            with self.subTest(base_url=base_url):
                requests = []

                def respond(request):
                    requests.append(request)
                    return httpx.Response(
                        200,
                        json={
                            "id": "test-completion",
                            "object": "chat.completion",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                            ],
                        },
                    )

                environment = {"MEMCO_LLM_BASE_URL": base_url, "MEMCO_LLM_API_KEY": api_key}
                with patch.dict(os.environ, environment, clear=True):
                    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
                        with patch(
                            "mas.llm_config.OpenAI",
                            side_effect=lambda **kwargs: OpenAI(http_client=http_client, **kwargs),
                        ):
                            with create_llm_client() as client:
                                response = client.chat.completions.create(
                                    model=model, messages=[{"role": "user", "content": "test"}]
                                )
                self.assertEqual("ok", response.choices[0].message.content)
                self.assertEqual(1, len(requests))
                self.assertEqual(base_url + "/chat/completions", str(requests[0].url))
                self.assertEqual(f"Bearer {api_key}", requests[0].headers["authorization"])
                self.assertEqual(model, json.loads(requests[0].content)["model"])


if __name__ == "__main__":
    unittest.main()
