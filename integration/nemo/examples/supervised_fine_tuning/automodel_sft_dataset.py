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
"""Instruction-following dataset factory for NeMo AutoModel SFT."""

from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def build_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def clean_response(response: str) -> str:
    return response.strip()


def _ensure_pad_token_id(tokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id

    eos_token = getattr(tokenizer, "eos_token", None)
    if getattr(tokenizer, "pad_token", None) is None and eos_token is not None:
        tokenizer.pad_token = eos_token
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return pad_token_id

    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    if callable(add_special_tokens):
        add_special_tokens({"pad_token": "<pad>"})
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return pad_token_id

    return getattr(tokenizer, "eos_token_id", 0)


def make_instruction_dataset(
    tokenizer,
    data_file: str,
    seq_length: int | None = None,
    limit_dataset_samples: int | None = None,
    use_chat_template: bool = False,
    fp8: bool = False,
    padding: bool = False,
    truncation: bool = True,
):
    """Return a NeMo AutoModel-compatible lazy mapped JSONL instruction dataset.

    The input JSONL must contain ``input`` and ``output`` fields. This matches the legacy SFT preprocessing utilities
    in this directory and the synthetic data generator used by the quickstart.
    """
    from datasets import load_dataset
    from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template, format_prompt_completion

    dataset = load_dataset("json", data_files=data_file, split="train")
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError(f"No SFT examples found in {data_file}.")

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if fp8 or pad_token_id is None:
        pad_token_id = _ensure_pad_token_id(tokenizer)

    def formatting_func(example):
        prompt = build_prompt(example["input"])
        answer = clean_response(example["output"])
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content": example["input"].strip()},
                {"role": "assistant", "content": answer},
            ]
            return format_chat_template(
                tokenizer=tokenizer,
                formatted_text=messages,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                seq_length=seq_length,
                padding=padding,
                truncation=truncation,
            )
        return format_prompt_completion(
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seq_length=seq_length,
            padding=padding,
            truncation=truncation,
            prompt=prompt,
            answer=answer,
        )

    return dataset.map(formatting_func, batched=False, remove_columns=dataset.column_names)
