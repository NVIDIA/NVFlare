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
"""Financial PhraseBank dataset factory for NeMo AutoModel PEFT."""

from __future__ import annotations


def _build_prompt(sentence: str) -> str:
    return f"{sentence} sentiment:"


def _clean_label(label: str) -> str:
    return f" {label.strip().lower()}"


def _balanced_indices(dataset, limit_dataset_samples: int) -> list[int]:
    """Build deterministic, label-balanced indices for small demo training windows."""
    buckets = {}
    for index, example in enumerate(dataset):
        label = _clean_label(example["label"])
        buckets.setdefault(label, []).append(index)

    selected = []
    offsets = {label: 0 for label in buckets}
    while len(selected) < limit_dataset_samples:
        added = False
        for label, indices in buckets.items():
            offset = offsets[label]
            if offset >= len(indices):
                continue
            selected.append(indices[offset])
            offsets[label] = offset + 1
            added = True
            if len(selected) >= limit_dataset_samples:
                break
        if not added:
            break
    return selected


def make_financial_phrase_dataset(
    tokenizer,
    data_file: str,
    seq_length: int | None = None,
    limit_dataset_samples: int | None = None,
    balance_labels: bool = False,
    use_chat_template: bool = False,
    fp8: bool = False,
    padding: bool = False,
    truncation: bool = True,
):
    """Return a NeMo AutoModel-compatible lazy mapped JSONL dataset.

    The input JSONL must contain ``sentence`` and ``label`` fields. Labels may use the legacy
    FinancialPhraseBank split format with a leading space, e.g. ``" positive"``.
    """
    from datasets import load_dataset
    from nemo_automodel.components.datasets.llm.formatting_utils import (
        _add_pad_token,
        format_chat_template,
        format_prompt_completion,
    )

    dataset = load_dataset("json", data_files=data_file, split="train")
    if limit_dataset_samples is not None:
        sample_count = min(limit_dataset_samples, len(dataset))
        if balance_labels:
            dataset = dataset.select(_balanced_indices(dataset, sample_count))
        else:
            dataset = dataset.select(range(sample_count))

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if fp8 or pad_token_id is None:
        added_pad_token_id = _add_pad_token(tokenizer)
        pad_token_id = (
            added_pad_token_id if added_pad_token_id is not None else getattr(tokenizer, "pad_token_id", None)
        )
    if pad_token_id is None:
        pad_token_id = eos_token_id

    def formatting_func(example):
        prompt = _build_prompt(example["sentence"])
        answer = _clean_label(example["label"])
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content": prompt},
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
