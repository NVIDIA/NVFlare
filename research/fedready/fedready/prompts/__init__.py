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

"""Centralized prompt template loading for FedReady agents."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

_PROMPT_FILES = {
    "server": "server.json",
    "client": "client.json",
}


@lru_cache(maxsize=None)
def _prompt_bundle(scope: str) -> dict[str, Any]:
    try:
        filename = _PROMPT_FILES[scope]
    except KeyError as exc:
        raise KeyError(f"unknown prompt scope: {scope}") from exc
    path = Path(__file__).with_name(filename)
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError(f"prompt bundle missing prompts object: {path}")
    rendered: dict[str, Any] = {}
    for key, value in prompts.items():
        if not isinstance(key, str):
            raise ValueError(f"prompt bundle has non-string key: {path}:{key}")
        rendered[key] = value
    return rendered


def render_prompt(scope: str, key: str, **values: Any) -> str:
    try:
        template = _prompt_bundle(scope)[key]
    except KeyError as exc:
        raise KeyError(f"missing prompt template: {scope}.{key}") from exc
    if not isinstance(template, str):
        raise TypeError(f"prompt template is not a string: {scope}.{key}")
    safe_values = {name: _stringify(value) for name, value in values.items()}
    return Template(template).safe_substitute(safe_values)


def render_prompt_object(scope: str, key: str, **values: Any) -> Any:
    try:
        template = deepcopy(_prompt_bundle(scope)[key])
    except KeyError as exc:
        raise KeyError(f"missing prompt template: {scope}.{key}") from exc
    safe_values = {name: _stringify(value) for name, value in values.items()}
    return _render_template_value(template, safe_values)


def render_server_prompt(key: str, **values: Any) -> str:
    return render_prompt("server", key, **values)


def render_client_prompt(key: str, **values: Any) -> str:
    return render_prompt("client", key, **values)


def render_server_prompt_object(key: str, **values: Any) -> Any:
    return render_prompt_object("server", key, **values)


def render_client_prompt_object(key: str, **values: Any) -> Any:
    return render_prompt_object("client", key, **values)


def _render_template_value(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(values)
    if isinstance(value, list):
        return [_render_template_value(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _render_template_value(item, values) for key, item in value.items()}
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
