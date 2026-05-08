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

import torch
from torch.optim import Optimizer


def _token_f1_score(pred: str, gt_answers: list[str]) -> float:
    try:
        from src.common import token_f1_score

        return token_f1_score(pred, gt_answers)
    except Exception:
        pred_tokens = pred.strip().lower().split()
        if not pred_tokens:
            return 0.0
        best = 0.0
        for answer in gt_answers:
            gt_tokens = str(answer).strip().lower().split()
            if not gt_tokens:
                continue
            common = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in set(gt_tokens))
            if common == 0:
                continue
            precision = common / len(pred_tokens)
            recall = common / len(gt_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
        return best


def _vlm_prompt_messages(example: dict) -> list[dict]:
    question = example["question"]
    prompt_prefix = example.get("prompt_prefix", "")
    system_message = example.get("system_message", "")
    text = f"{question}\n{prompt_prefix}" if prompt_prefix else question
    messages: list[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    )
    return messages


def evaluate_vlm_generative(
    model,
    processor,
    dataset,
    *,
    batch_size: int,
    max_new_tokens: int,
    audit_samples: int = 0,
    audit_prefix: str = "vlm_eval",
    device=None,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(dataset) == 0:
        raise ValueError("evaluate_vlm_generative() called with an empty dataset.")

    model.eval()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    prior_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    f1_sum = 0.0
    n_seen = 0
    try:
        with torch.no_grad():
            for offset in range(0, len(dataset), batch_size):
                batch = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
                prompt_texts = [
                    processor.apply_chat_template(
                        _vlm_prompt_messages(example),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for example in batch
                ]
                images = [example["image"] for example in batch]
                inputs = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    do_sample=False,
                )
                trimmed = [out_ids[input_len:] for out_ids in generated_ids]
                preds = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for example, pred, trimmed_ids in zip(batch, preds, trimmed):
                    pred = pred.strip()
                    f1 = _token_f1_score(pred, example["answers"])
                    f1_sum += f1
                    n_seen += 1
                    if audit_samples > 0 and n_seen <= audit_samples:
                        print(
                            f"{audit_prefix}: sample={n_seen} "
                            f"dataset={example.get('dataset_name', '')} "
                            f"gen_tokens={len(trimmed_ids)} token_f1={f1:.4f}"
                        )
                        print(f"{audit_prefix}: question={example.get('question', '')}")
                        print(f"{audit_prefix}: gt_primary={example.get('gt_primary', '')}")
                        print(f"{audit_prefix}: prediction={pred!r}")
    finally:
        processor.tokenizer.padding_side = prior_padding_side

    if n_seen == 0:
        raise ValueError("evaluate_vlm_generative() processed zero examples.")
    return f1_sum / n_seen


def compute_model_diff(model, global_model):
    local_weights = model.state_dict()
    global_weights = global_model.state_dict()
    missing_params = []
    model_diff = {}
    diff_norm = 0.0

    for name in global_weights:
        if name not in local_weights:
            missing_params.append(name)
            continue
        model_diff[name] = (local_weights[name] - global_weights[name]).cpu()
        diff_norm += torch.linalg.norm(model_diff[name])

    if len(model_diff) == 0 or len(missing_params) > 0:
        raise ValueError(f"No weight differences computed or missing parameters: {missing_params}")

    if torch.isnan(diff_norm) or torch.isinf(diff_norm):
        raise ValueError(f"Diff norm is NaN or Inf: {diff_norm}")

    return model_diff, diff_norm


def get_lr_values(optimizer: Optimizer):
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]
