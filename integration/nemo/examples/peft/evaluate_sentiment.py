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
"""Evaluate Financial PhraseBank sentiment predictions for Nemotron 3 PEFT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict

import adapter_checkpoint
import model_profiles
import torch

LABELS = ("neutral", "positive", "negative")
DEFAULT_PROMPT_TEMPLATE = "{sentence} sentiment:"
DEFAULT_CHOICE_MAP = "neutral=neutral,positive=positive,negative=negative"


def define_parser():
    parser = argparse.ArgumentParser(description="Evaluate sentiment labels by exact label log-probability scoring.")
    model_profiles.add_model_profile_argument(parser)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument(
        "--validation_file",
        default="./data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl",
    )
    parser.add_argument(
        "--test_file",
        default="./data/FinancialPhraseBank-v1.0/financial_phrase_bank_test.jsonl",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Evaluate validation only; use this for intermediate rounds to keep the test split untouched.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--choice_map", default=DEFAULT_CHOICE_MAP)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device_map", default="auto")
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
    parser.add_argument("--positive_bias", type=float, default=0.0)
    parser.add_argument("--negative_bias", type=float, default=0.0)
    parser.add_argument(
        "--search_validation_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search positive/negative score biases on validation and report the resulting test score.",
    )
    parser.add_argument("--bias_search_max", type=float, default=12.0)
    parser.add_argument("--bias_search_step", type=float, default=0.1)
    args = model_profiles.resolve_model_profile(parser.parse_args())
    if args.output_dir is None:
        suffix = (
            "nemotron35_lightning_exact_eval" if model_profiles.is_lightning35(args) else "nemotron3_nano_exact_eval"
        )
        args.output_dir = os.path.join(".", "models", suffix)
    return args


def parse_choice_map(value: str) -> dict[str, str]:
    mapping = {}
    for item in value.split(","):
        label, choice = item.split("=", 1)
        label = label.strip()
        choice = choice.strip()
        if label not in LABELS:
            raise ValueError(f"Unknown label in choice map: {label}")
        mapping[label] = choice
    missing = set(LABELS) - set(mapping)
    if missing:
        raise ValueError(f"Missing labels in choice map: {sorted(missing)}")
    return mapping


def load_rows(path: str) -> list[dict[str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            rows.append({"sentence": item["sentence"], "expected": item["label"].strip()})
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def score_rows(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    batch_size: int,
    prompt_template: str,
    choice_map: dict[str, str],
) -> list[dict[str, float]]:
    candidates_by_len = defaultdict(list)
    scores = [dict() for _ in rows]
    for row_idx, row in enumerate(rows):
        prompt = prompt_template.format(sentence=row["sentence"])
        prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        for label in LABELS:
            choice = choice_map[label]
            input_ids = tokenizer(f"{prompt} {choice}", return_tensors="pt", add_special_tokens=False).input_ids
            response_token_count = int(input_ids.shape[1]) - prompt_len
            candidates_by_len[int(input_ids.shape[1])].append(
                {
                    "row_idx": row_idx,
                    "label": label,
                    "choice": choice,
                    "prompt_len": prompt_len,
                    "input_ids": input_ids.squeeze(0),
                    "response_token_count": response_token_count,
                }
            )

    with torch.no_grad():
        for candidates in candidates_by_len.values():
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start : start + batch_size]
                input_ids = torch.stack([item["input_ids"] for item in batch]).to(model.device)
                outputs = model(input_ids)
                logits = outputs.logits.float()
                for batch_idx, item in enumerate(batch):
                    prompt_len = item["prompt_len"]
                    seq_len = int(item["input_ids"].shape[0])
                    target = input_ids[batch_idx, prompt_len:seq_len]
                    label_logits = logits[batch_idx, prompt_len - 1 : seq_len - 1, :]
                    log_probs = torch.log_softmax(label_logits, dim=-1)
                    token_scores = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    scores[item["row_idx"]][item["label"]] = float(token_scores.sum().item())
                    if item["label"] == rows[item["row_idx"]]["expected"]:
                        scores[item["row_idx"]]["__gold_response_token_count"] = item["response_token_count"]
    return scores


def apply_bias(
    row_scores: list[dict[str, float]], positive_bias: float, negative_bias: float
) -> list[dict[str, float]]:
    biased_scores = []
    for scores in row_scores:
        adjusted = dict(scores)
        adjusted["positive"] += positive_bias
        adjusted["negative"] += negative_bias
        biased_scores.append(adjusted)
    return biased_scores


def summarize(rows: list[dict[str, str]], row_scores: list[dict[str, float]]) -> dict:
    predictions = []
    for idx, (row, scores) in enumerate(zip(rows, row_scores)):
        prediction = max(LABELS, key=scores.get)
        gold_score = scores[row["expected"]]
        gold_token_count = int(scores.get("__gold_response_token_count", 0))
        predictions.append(
            {
                "index": idx,
                "sentence": row["sentence"],
                "expected": row["expected"],
                "prediction": prediction,
                "match": prediction == row["expected"],
                "neutral_score": scores["neutral"],
                "positive_score": scores["positive"],
                "negative_score": scores["negative"],
                "margin": scores[prediction] - max(scores[label] for label in LABELS if label != prediction),
                "gold_log_probability": gold_score,
                "gold_response_token_count": gold_token_count,
            }
        )

    counts = Counter(row["expected"] for row in predictions)
    pred_counts = Counter(row["prediction"] for row in predictions)
    confusion = {expected: {pred: 0 for pred in LABELS} for expected in LABELS}
    for row in predictions:
        confusion[row["expected"]][row["prediction"]] += 1

    per_label = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "support": counts[label],
            "predicted": pred_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(row["match"] for row in predictions)
    response_token_count = sum(row["gold_response_token_count"] for row in predictions)
    response_token_nll = -sum(row["gold_log_probability"] for row in predictions)
    return {
        "total": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "macro_f1": sum(f1s) / len(f1s),
        "response_token_loss": response_token_nll / response_token_count if response_token_count else None,
        "response_token_count": response_token_count,
        "label_counts": dict(counts),
        "prediction_counts": dict(pred_counts),
        "confusion": confusion,
        "per_label": per_label,
        "predictions": predictions,
    }


def write_predictions_csv(summary: dict, csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    predictions = summary.get("predictions", [])
    if not predictions:
        raise ValueError("No predictions to write.")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)


def eval_split(
    name: str,
    path: str,
    output_dir: str,
    model,
    tokenizer,
    batch_size: int,
    prompt_template: str,
    choice_map: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, float]], dict]:
    rows = load_rows(path)
    scores = score_rows(model, tokenizer, rows, batch_size, prompt_template, choice_map)
    summary = summarize(rows, scores)
    csv_path = os.path.join(output_dir, f"{name}_predictions.csv")
    write_predictions_csv(summary, csv_path)
    summary = {k: v for k, v in summary.items() if k != "predictions"}
    summary["csv_path"] = csv_path
    return rows, scores, summary


def best_bias(
    rows: list[dict[str, str]],
    row_scores: list[dict[str, float]],
    max_bias: float,
    step: float,
) -> dict:
    best = None
    steps = int(round(max_bias / step))
    for positive_idx in range(steps + 1):
        positive = round(positive_idx * step, 10)
        for negative_idx in range(steps + 1):
            negative = round(negative_idx * step, 10)
            summary = summarize(rows, apply_bias(row_scores, positive, negative))
            if best is None or summary["macro_f1"] > best["summary"]["macro_f1"]:
                best = {"biases": {"positive": positive, "negative": negative}, "summary": summary}
    return best


def _summary_without_predictions(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k != "predictions"}


def load_model(args):
    if model_profiles.is_lightning35(args):
        return _load_lightning_model(args)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
    return model, tokenizer


def _split_modules(value: str) -> list[str]:
    if not value or value == "all-linear":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _lightning_profile_settings(args) -> dict:
    return model_profiles.adapter_compatibility_settings(args)


def _load_lightning_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The lightning35 native backend requires a CUDA GPU.")
    from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoTokenizer
    from nemo_automodel.components._peft.lora import PeftConfig
    from nemo_automodel.components.checkpoint import CheckpointingConfig
    from nemo_automodel.components.models.common import BackendConfig

    tokenizer = NeMoAutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    peft_config = None
    if args.adapter_dir:
        peft_config = PeftConfig(
            target_modules=_split_modules(args.target_modules),
            exclude_modules=_split_modules(args.exclude_modules),
            match_all_linear=args.target_modules == "all-linear" and not args.exclude_modules,
            dim=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            use_triton=args.use_triton_lora,
        )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        peft_config=peft_config,
        num_nextn_predict_layers=2,
        mtp_use_repeated_layer=True,
        mtp_loss_scaling_factor=0.1,
        backend=BackendConfig(
            attn="te",
            linear="torch",
            rms_norm="torch_fp32",
            experts="torch_mm",
            dispatcher="torch",
        ),
    )
    if args.adapter_dir:
        incoming_state = adapter_checkpoint.strip_model_prefix(adapter_checkpoint.load_adapter_state(args.adapter_dir))
        manifest = adapter_checkpoint.load_adapter_manifest(args.adapter_dir)
        adapter_checkpoint.validate_adapter_manifest(
            manifest,
            incoming_state,
            expected={
                "model_profile": args.model_profile,
                "base_model_name_or_path": args.model_name_or_path,
                "base_model_revision": args.model_revision,
                "tokenizer_name_or_path": args.tokenizer_name_or_path,
                "tokenizer_revision": args.tokenizer_revision,
                "profile_settings": _lightning_profile_settings(args),
            },
        )
        with tempfile.TemporaryDirectory(prefix="nvflare_lightning35_eval_") as temp_dir:
            adapter_checkpoint.save_hf_adapter_state_dir(
                incoming_state,
                temp_dir,
                adapter_manifest=manifest,
            )
            checkpointer = CheckpointingConfig(is_peft=True).build(dp_rank=0, tp_rank=0, pp_rank=0)
            checkpointer.load_model(model, temp_dir)
        unexpected_trainable = [
            name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name
        ]
        if unexpected_trainable:
            raise RuntimeError(f"Base parameters are trainable during evaluation: {unexpected_trainable[:10]}")
    model.eval()
    return model, tokenizer


def main():
    args = define_parser()
    choice_map = parse_choice_map(args.choice_map)
    model, tokenizer = load_model(args)

    val_rows, val_scores, val_summary = eval_split(
        "validation",
        args.validation_file,
        args.output_dir,
        model,
        tokenizer,
        args.batch_size,
        args.prompt_template,
        choice_map,
    )
    test_rows = test_scores = test_summary = None
    if not args.validation_only:
        test_rows, test_scores, test_summary = eval_split(
            "test",
            args.test_file,
            args.output_dir,
            model,
            tokenizer,
            args.batch_size,
            args.prompt_template,
            choice_map,
        )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "model_profile": args.model_profile,
        "model_revision": args.model_revision,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_revision": args.tokenizer_revision,
        "adapter_dir": args.adapter_dir,
        "prompt_template": args.prompt_template,
        "choice_map": choice_map,
        "scoring": "padding_free_grouped_by_exact_sequence_length",
        "validation": val_summary,
    }
    if test_summary is not None:
        summary["test"] = test_summary

    if args.positive_bias or args.negative_bias:
        summary["fixed_bias"] = {
            "biases": {"positive": args.positive_bias, "negative": args.negative_bias},
            "validation": _summary_without_predictions(
                summarize(val_rows, apply_bias(val_scores, args.positive_bias, args.negative_bias))
            ),
        }
        if test_rows is not None:
            summary["fixed_bias"]["test"] = _summary_without_predictions(
                summarize(test_rows, apply_bias(test_scores, args.positive_bias, args.negative_bias))
            )

    if args.search_validation_bias:
        bias = best_bias(val_rows, val_scores, args.bias_search_max, args.bias_search_step)
        summary["best_validation_bias"] = {
            "biases": bias["biases"],
            "validation": _summary_without_predictions(bias["summary"]),
        }
        if test_rows is not None:
            summary["best_validation_bias"]["test"] = _summary_without_predictions(
                summarize(
                    test_rows,
                    apply_bias(test_scores, bias["biases"]["positive"], bias["biases"]["negative"]),
                )
            )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
