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
"""Generate text from a federated SFT full-model checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import model_checkpoint
import torch
from automodel_sft_dataset import build_prompt

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_PROMPTS = [
    "Explain federated learning in one sentence.",
    "What does a federated server aggregate?",
]


def define_parser():
    parser = argparse.ArgumentParser(description="Run generation from a federated SFT checkpoint.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", default="./models/nemotron3_nano_sft_predictions.json")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def _compatible_state(model_state, checkpoint_state):
    compatible = OrderedDict()
    for key, target in model_state.items():
        value = checkpoint_state.get(key)
        if isinstance(value, torch.Tensor) and value.shape == target.shape:
            compatible[key] = value.detach().to(device=target.device, dtype=target.dtype)
    return compatible


def _greedy_generate_no_cache(model, input_ids, max_new_tokens: int, eos_token_id: int | None = None):
    generated = input_ids
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        outputs = model(input_ids=generated, attention_mask=attention_mask, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
    return generated


def _write_predictions(output_json: str, checkpoint: str, results: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"checkpoint": checkpoint, "predictions": results}, f, indent=2)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = define_parser()
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    checkpoint_state = model_checkpoint.load_model_state(args.checkpoint)
    compatible = _compatible_state(model.state_dict(), checkpoint_state)
    if not compatible:
        raise RuntimeError(f"No checkpoint tensors from {args.checkpoint} matched the model state dict.")
    model.load_state_dict(compatible, strict=False)
    model.eval()

    prompts = args.prompt or DEFAULT_PROMPTS
    results = []
    for prompt in prompts:
        formatted_prompt = build_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = _greedy_generate_no_cache(
                model,
                input_ids=inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        results.append({"prompt": prompt, "generated": generated})

    _write_predictions(args.output_json, args.checkpoint, results)
    for item in results:
        print(f"PROMPT: {item['prompt']}")
        print(f"GENERATED: {item['generated']}")


if __name__ == "__main__":
    main()
