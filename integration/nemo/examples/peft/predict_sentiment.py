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
"""Run notebook sentiment predictions from a federated Nemotron 3 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os

import adapter_checkpoint
import model_profiles
import torch

SENTIMENT_LABELS = ("neutral", "positive", "negative")
NOTEBOOK_EXAMPLES = [
    ("The products have a low salt and fat content . sentiment:", "neutral"),
    ("The agreement is valid for four years . sentiment:", "neutral"),
    ("Diluted EPS rose to EUR3 .68 from EUR0 .50 . sentiment:", "positive"),
    (
        "Profit before taxes decreased by 9 % to EUR 187.8 mn in the first nine months of 2008 , "
        "compared to EUR 207.1 mn a year earlier . sentiment:",
        "negative",
    ),
]


def define_parser():
    parser = argparse.ArgumentParser(description="Score notebook sentiment prompts with a federated LoRA adapter.")
    model_profiles.add_model_profile_argument(parser)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--server_model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", default=None)
    parser.add_argument("--exclude_modules", default=None)
    parser.add_argument("--use_triton_lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--cp_size", type=int, default=None)
    parser.add_argument("--ep_size", type=int, default=None)
    parser.add_argument("--activation_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = model_profiles.resolve_model_profile(parser.parse_args())
    if args.server_model is None:
        args.server_model = model_profiles.default_server_model_path(args)
    if args.output_dir is None:
        name = "nemotron35_lightning_lora_final" if model_profiles.is_lightning35(args) else "nemotron3_nano_lora_final"
        args.output_dir = os.path.join(".", "models", name)
    if args.output_json is None:
        name = (
            "nemotron35_lightning_prediction_summary.json"
            if model_profiles.is_lightning35(args)
            else "nemotron3_nano_prediction_summary.json"
        )
        args.output_json = os.path.join(".", "models", name)
    return args


def _split_target_modules(target_modules: str):
    if target_modules == "all-linear":
        return target_modules
    return [item.strip() for item in target_modules.split(",") if item.strip()]


def _adapter_config(args) -> dict:
    return {
        "base_model_name_or_path": args.model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "peft_type": "LORA",
        "r": args.lora_rank,
        "target_modules": _split_target_modules(args.target_modules),
        "task_type": "CAUSAL_LM",
    }


def _prepare_hf_adapter_dir(args) -> tuple[str, int]:
    adapter_state = adapter_checkpoint.strip_model_prefix(adapter_checkpoint.load_adapter_state(args.server_model))
    adapter_checkpoint.save_hf_adapter_state_dir(
        adapter_state,
        args.output_dir,
        adapter_config=_adapter_config(args),
    )
    return args.output_dir, len(adapter_state)


def _score_labels(model, tokenizer, prompt: str) -> dict[str, float]:
    scores = {}
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]
        for label in SENTIMENT_LABELS:
            ids = tokenizer(f"{prompt} {label}", return_tensors="pt", add_special_tokens=False).input_ids.to(
                model.device
            )
            outputs = model(ids)
            logits = outputs.logits[:, prompt_len - 1 : -1, :]
            target = ids[:, prompt_len:]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scores[label] = float(token_scores.sum().item())
    return scores


def classify(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def predict(args):
    if model_profiles.is_lightning35(args):
        import evaluate_sentiment

        adapter_dir = args.server_model
        adapter_tensor_count = len(adapter_checkpoint.load_adapter_state(args.server_model))
        args.adapter_dir = args.server_model
        model, tokenizer = evaluate_sentiment.load_model(args)
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_dir, adapter_tensor_count = _prepare_hf_adapter_dir(args)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            revision=args.tokenizer_revision,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        base_model.eval()
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()

    rows = []
    for prompt, expected in NOTEBOOK_EXAMPLES:
        scores = _score_labels(model, tokenizer, prompt)
        prediction = classify(scores)
        rows.append(
            {
                "prompt": prompt,
                "expected": expected,
                "prediction": prediction,
                "match": prediction == expected,
                "scores": scores,
            }
        )

    return {
        "server_model": args.server_model,
        "adapter_dir": adapter_dir,
        "adapter_tensor_count": adapter_tensor_count,
        "results": rows,
        "all_match": all(row["match"] for row in rows),
    }


def main():
    args = define_parser()
    summary = predict(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("The prediction results of some sample queries with the trained model:")
    for row in summary["results"]:
        print(f"{row['prompt']} {row['prediction']}")
    print(f"All predictions match expected labels: {summary['all_match']}")
    print(f"Prediction summary saved to: {args.output_json}")


if __name__ == "__main__":
    main()
