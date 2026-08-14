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

"""Model loading, text formatting, and state helpers for federated SFT."""

import torch


def format_example(example: dict) -> str:
    """Convert supported instruction records to the text consumed by TRL."""
    if example.get("text"):
        return example["text"]
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def precision_config(precision: str = "auto") -> tuple[bool, bool, torch.dtype]:
    """Resolve the requested trainer flags and model-loading dtype."""
    if precision not in ("auto", "float32", "bfloat16"):
        raise ValueError(f"unsupported precision: {precision}")
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if precision == "bfloat16" and not bf16_supported:
        raise RuntimeError("bfloat16 precision requires a CUDA device with BF16 support")
    if precision == "bfloat16" or (precision == "auto" and bf16_supported):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return dtype == torch.bfloat16, False, dtype


def load_model_and_tokenizer(
    model_name_or_path: str,
    model_revision: str | None,
    trust_remote_code: bool,
    precision: str,
):
    """Load matching Hugging Face model and tokenizer objects for one client."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, _, dtype = precision_config(precision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    return model, tokenizer


def cpu_model_state(model) -> dict[str, torch.Tensor]:
    """Copy the full model state to CPU for transport through Collab."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
