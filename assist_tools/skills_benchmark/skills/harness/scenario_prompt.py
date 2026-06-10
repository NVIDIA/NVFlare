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

"""Prompt path, template, and materialization helpers for benchmark scenarios."""

from __future__ import annotations

import hashlib
import re
import string
from pathlib import Path
from typing import Any, Mapping

from .scenario_common import ScenarioValidationError, require_non_empty_string, resolve_path, slugify

MAX_PROMPT_BYTES = 4 * 1024 * 1024
GENERATED_PROMPT_DIR = ".agent_benchmark/rendered_prompts"
PROMPT_TEMPLATE_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROMPT_TEMPLATE_SCALAR_TYPES = (str, int, float, bool)
PROMPT_TEMPLATE_LITERAL_BRACE_HINT = "Use '{{' and '}}' for literal braces in prompt templates."


def prompt_template_fields(template_text: str) -> set[str]:
    fields = set()
    try:
        parsed = list(string.Formatter().parse(template_text))
    except ValueError as exc:
        raise ScenarioValidationError(
            f"Prompt template is invalid: {exc}. {PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
        ) from exc
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name == "":
            raise ScenarioValidationError(
                f"Prompt templates must not use positional '{{}}' placeholders. {PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
            )
        if "." in field_name or "[" in field_name or "]" in field_name:
            raise ScenarioValidationError("Prompt templates must not use attribute or index access")
        if not PROMPT_TEMPLATE_VARIABLE_PATTERN.match(field_name):
            raise ScenarioValidationError(
                f"Prompt template placeholder has an unsafe name: {field_name}. "
                f"{PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
            )
        if format_spec or conversion:
            raise ScenarioValidationError(
                f"Prompt templates must not use format specifiers or conversions. "
                f"{PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
            )
        fields.add(field_name)
    return fields


def prompt_template_variables(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ScenarioValidationError("prompt.variables must be a mapping")
    variables: dict[str, str] = {}
    for key, raw_value in value.items():
        name = require_non_empty_string(key, "prompt variable name")
        if not PROMPT_TEMPLATE_VARIABLE_PATTERN.match(name):
            raise ScenarioValidationError(f"prompt.variables.{name} has an unsafe variable name")
        if raw_value is None or not isinstance(raw_value, PROMPT_TEMPLATE_SCALAR_TYPES):
            raise ScenarioValidationError(f"prompt.variables.{name} must be a scalar string, number, or boolean")
        variables[name] = str(raw_value)
    return variables


def render_prompt_template(template_text: str, variables: Mapping[str, str]) -> str:
    fields = prompt_template_fields(template_text)
    missing = fields.difference(variables)
    if missing:
        raise ScenarioValidationError(
            f"Prompt template missing variable(s): {', '.join(sorted(missing))}. "
            f"{PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
        )
    unused = set(variables).difference(fields)
    if unused:
        raise ScenarioValidationError(f"prompt.variables contains unused variable(s): {', '.join(sorted(unused))}")
    try:
        return template_text.format(**variables)
    except (KeyError, IndexError, ValueError) as exc:
        raise ScenarioValidationError(
            f"Prompt template could not be rendered: {exc}. {PROMPT_TEMPLATE_LITERAL_BRACE_HINT}"
        ) from exc


def read_prompt_bytes(prompt_path: Path) -> bytes:
    try:
        prompt_bytes = prompt_path.read_bytes()
    except OSError as exc:
        raise ScenarioValidationError(f"Could not read prompt file {prompt_path}: {exc}") from exc
    if len(prompt_bytes) > MAX_PROMPT_BYTES:
        raise ScenarioValidationError(
            f"Prompt file exceeds max size {MAX_PROMPT_BYTES} bytes: {prompt_path} ({len(prompt_bytes)} bytes)"
        )
    return prompt_bytes


def resolve_prompt_path(prompt_text: str, base_dir: Path, *, allow_external_prompt: bool) -> Path:
    prompt_path = resolve_path(prompt_text, base_dir)
    base_root = base_dir.resolve()
    resolved_prompt_path = prompt_path.resolve()
    if not allow_external_prompt and not resolved_prompt_path.is_relative_to(base_root):
        raise ScenarioValidationError(f"Prompt file must stay within scenario directory {base_root}: {prompt_path}")
    if not prompt_path.is_file():
        raise ScenarioValidationError(f"Prompt file must exist: {prompt_path}")
    return resolved_prompt_path


def rendered_prompt_filename(source_path: Path, rendered_bytes: bytes) -> str:
    rendered_hash = hashlib.sha256(rendered_bytes).hexdigest()
    return f"{slugify(source_path.stem)}_{rendered_hash[:12]}.txt"


def contains_inline_template_placeholder(value: str) -> bool:
    try:
        return any(
            field_name is not None and PROMPT_TEMPLATE_VARIABLE_PATTERN.match(field_name)
            for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(value)
        )
    except ValueError:
        return False


def materialize_prompt_for_output(scenario: dict[str, Any], run_plan: dict[str, Any], output_dir: Path) -> None:
    prompt = scenario.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("source_type") != "template":
        return
    rendered_text = prompt.pop("_rendered_text", None)
    filename = prompt.pop("_rendered_filename", None)
    if not isinstance(rendered_text, str) or not isinstance(filename, str):
        return
    rendered_bytes = rendered_text.encode("utf-8")
    prompt_dir = output_dir / GENERATED_PROMPT_DIR
    prompt_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = (prompt_dir / filename).resolve()
    rendered_path.write_bytes(rendered_bytes)
    prompt["path"] = str(rendered_path)
    prompt["rendered_path"] = str(rendered_path)
    for entry in run_plan.get("entries") or []:
        if isinstance(entry, dict):
            entry["prompt_source"] = str(rendered_path)


def resolve_prompt(raw: Mapping[str, Any], base_dir: Path, *, allow_external_prompt: bool = False) -> dict[str, Any]:
    value = raw.get("prompt_path") or raw.get("prompt_file") or raw.get("prompt")
    source_type = "file"
    variables: dict[str, str] = {}
    if isinstance(value, dict):
        prompt_mapping = value
        variables = prompt_template_variables(prompt_mapping.get("variables"))
        if prompt_mapping.get("template") and prompt_mapping.get("path"):
            raise ScenarioValidationError("prompt must not set both template and path")
        if prompt_mapping.get("template"):
            source_type = "template"
            value = prompt_mapping.get("template")
        elif prompt_mapping.get("path"):
            value = prompt_mapping.get("path")
            source_type = "template" if variables else "file"
        else:
            raise ScenarioValidationError("prompt mapping must contain path or template")
    prompt_text = require_non_empty_string(value, "prompt path")
    if source_type == "file" and contains_inline_template_placeholder(prompt_text):
        raise ScenarioValidationError(
            "prompt must be a file path, not an inline template string. "
            "Use a prompt template file with prompt.template or prompt.path plus prompt.variables."
        )
    resolved_prompt_path = resolve_prompt_path(prompt_text, base_dir, allow_external_prompt=allow_external_prompt)
    source_bytes = read_prompt_bytes(resolved_prompt_path)
    prompt_bytes = source_bytes
    prompt_path = resolved_prompt_path
    if source_type == "template":
        try:
            template_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScenarioValidationError(f"Prompt template must be UTF-8 text: {resolved_prompt_path}") from exc
        rendered_text = render_prompt_template(template_text, variables)
        prompt_bytes = rendered_text.encode("utf-8")
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            raise ScenarioValidationError(
                f"Rendered prompt exceeds max size {MAX_PROMPT_BYTES} bytes: "
                f"{resolved_prompt_path} ({len(prompt_bytes)} bytes)"
            )
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    prompt = {
        "path": str(prompt_path),
        "source_path": str(resolved_prompt_path),
        "source_type": source_type,
        "sha256": prompt_sha,
        "bytes": len(prompt_bytes),
        "rendered_sha256": prompt_sha,
        "rendered_bytes": len(prompt_bytes),
        "source_sha256": source_sha,
        "source_bytes": len(source_bytes),
    }
    if source_type == "template":
        prompt["_rendered_text"] = prompt_bytes.decode("utf-8")
        prompt["_rendered_filename"] = rendered_prompt_filename(resolved_prompt_path, prompt_bytes)
    return prompt
