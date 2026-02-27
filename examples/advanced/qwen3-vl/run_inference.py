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
"""
Short inference script for Qwen3-VL on PubMedVision-style samples.
Use to compare base vs fine-tuned checkpoints (e.g. before/after federated SFT).

Example:
  # Base model (no fine-tuning; uses ./data/site-1/train.json and ./PubMedVision, max 1 sample by default)
  python run_inference.py --model_path Qwen/Qwen3-VL-2B-Instruct

  # Fine-tuned checkpoint (saved by FL job or client)
  python run_inference.py --model_path ./path/to/checkpoint-xxx

  # NVFlare global model (single .pt file from server; base model defaults to Qwen/Qwen3-VL-2B-Instruct)
  python run_inference.py --model_path /path/to/FL_global_model.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Use example's model loader (supports both Qwen2.5-VL and Qwen3-VL)
from model import load_qwen_vl_from_pretrained
from transformers import AutoProcessor

# Key used by NVFlare PT persistor when saving FL_global_model.pt
NVFLARE_PT_MODEL_KEY = "model"


def _ensure_tensors(state_dict: dict) -> dict:
    """Convert numpy values to torch tensors so load_state_dict works correctly."""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):  # numpy array
            out[k] = torch.as_tensor(v, device="cpu")
        else:
            out[k] = v
    return out


def _align_ckpt_to_model(state_dict: dict, model_keys: set) -> dict:
    """Map checkpoint keys to model keys. FL saves wrapper state_dict ('model.xxx' or 'model.model.xxx')."""
    if not state_dict or not model_keys:
        return dict(state_dict)

    # Try mappings and pick the one that matches the most model keys
    def strip_one(k):
        return k.replace("model.", "", 1) if k.startswith("model.") else k

    def strip_two(k):
        if k.startswith("model.model."):
            return k.replace("model.model.", "", 1)
        return k

    candidates = [
        state_dict,  # as-is
        {strip_one(k): v for k, v in state_dict.items()},  # strip one "model."
        {strip_two(k): v for k, v in state_dict.items()},  # strip "model.model."
    ]
    best = max(candidates, key=lambda d: len(model_keys & set(d.keys())))
    return best


def _load_nvflare_global_pt(pt_path: str, model_keys: set) -> dict:
    """Load state dict from NVFlare FL_global_model.pt and align keys to the inner HuggingFace model."""
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {pt_path}, got {type(data)}")
    state_dict = data.get(NVFLARE_PT_MODEL_KEY)
    if state_dict is None:
        raise ValueError(f"No key {NVFLARE_PT_MODEL_KEY!r} in {pt_path}. Keys: {list(data.keys())}")
    state_dict = _ensure_tensors(state_dict)
    state_dict = _align_ckpt_to_model(state_dict, model_keys)
    return state_dict


def _abs_image_path(image_path: str, image_root: str) -> str:
    """Resolve image path; if relative, join with image_root."""
    path = image_path.strip()
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    full = os.path.join(image_root, path)
    if os.path.isfile(full):
        return os.path.abspath(full)
    return os.path.abspath(path)


def load_samples(data_file: str, image_root: str, max_samples: int):
    """Load PubMedVision-style records (one or more images + conversations with from/value)."""
    with open(data_file, "r") as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = [records]
    samples = []
    for r in records[:max_samples]:
        images = r.get("image")
        if images is None:
            images = r.get("images")
        if isinstance(images, str):
            images = [images]
        if not images:
            continue
        convs = r.get("conversations", [])
        question = None
        answer = None
        for turn in convs:
            if turn.get("from") == "human":
                question = turn.get("value", "")
                break
        for turn in convs:
            if turn.get("from") == "gpt":
                answer = turn.get("value", "")
                break
        if question is None:
            continue
        image_paths = []
        missing = False
        for image in images:
            img_abs = _abs_image_path(image, image_root)
            if not os.path.isfile(img_abs):
                print(f"Warning: image not found {img_abs}", file=sys.stderr)
                missing = True
                break
            image_paths.append(img_abs)
        if missing:
            continue
        samples.append({"image_paths": image_paths, "question": question, "ground_truth": answer or ""})
    return samples


def build_messages(image_paths, question: str):
    """Build chat messages for Qwen: one user turn with one or more images + text."""
    content = [{"type": "image", "image": image_path} for image_path in image_paths]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL inference on PubMedVision-style data")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID or path to checkpoint (base or fine-tuned)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/site-1/train.json",
        help="Path to PubMedVision-style JSON (default: ./data/site-1/train.json, after prepare_data.py)",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="./PubMedVision",
        help="Root to resolve relative image paths (default: ./PubMedVision, after download_data.py)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1,
        help="Max number of samples to run (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens per reply (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID for architecture and processor when --model_path is an NVFlare .pt file (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    args = parser.parse_args()

    image_root = args.image_root or os.path.dirname(os.path.abspath(args.data_file))

    is_nvflare_pt = os.path.isfile(args.model_path) and args.model_path.endswith(".pt")
    if is_nvflare_pt:
        base_path = args.base_model
        print(f"Loading base model and processor from: {base_path}")
        model = load_qwen_vl_from_pretrained(base_path, dtype=torch.bfloat16)
        model_keys = set(model.state_dict().keys())
        # Snapshot one weight from base to detect if .pt is identical to base (global never updated)
        _sample_key = next((k for k in model_keys if "weight" in k and "embed" in k), next(iter(model_keys)))
        base_sample = model.state_dict()[_sample_key].clone()
        print(f"Loading NVFlare global weights from: {args.model_path}")
        state_dict = _load_nvflare_global_pt(args.model_path, model_keys)
        result = model.load_state_dict(state_dict, strict=False)
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        n_applied = len(model_keys) - n_missing
        print(f"Checkpoint: applied {n_applied} keys; missing_keys={n_missing}, unexpected_keys={n_unexpected}")
        ckpt_sample = model.state_dict()[_sample_key]
        if torch.equal(base_sample, ckpt_sample):
            print(
                "  Note: checkpoint weight sample matches base model (key=%s). "
                "FL_global_model.pt may be the initial save or the global model was never updated." % (_sample_key,)
            )
        else:
            diff = (ckpt_sample.float() - base_sample.float()).abs().max().item()
            print(f"  Checkpoint differs from base (max abs diff on sample key {_sample_key!r}: {diff:.2e})")
        if n_applied == 0:
            sample_ckpt = list(state_dict.keys())[:3]
            sample_model = list(model_keys)[:3]
            print(f"  Checkpoint key sample: {sample_ckpt}")
            print(f"  Model key sample:      {sample_model}")
            raise RuntimeError(
                "No checkpoint keys matched the model. Check that --base_model matches the model "
                "used in the FL job (e.g. Qwen/Qwen3-VL-2B-Instruct)."
            )
        if n_missing > 0 and n_missing <= 5:
            print(f"  missing_keys: {result.missing_keys}")
        elif n_missing > 5:
            print(f"  missing_keys (first 5): {result.missing_keys[:5]}")
        model = model.to(args.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    else:
        print(f"Loading model and processor from: {args.model_path}")
        model = load_qwen_vl_from_pretrained(args.model_path, dtype=torch.bfloat16)
        model = model.to(args.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if args.data_file and os.path.isfile(args.data_file):
        samples = load_samples(args.data_file, image_root, args.max_samples)
        if not samples:
            print("No valid samples found. Check --data_file and --image_root.")
            return 1
        print(f"Running inference on {len(samples)} sample(s)\n")
    else:
        # Single prompt mode: no data file, optional manual test
        print(
            "Data file not found. Run download_data.py and prepare_data.py first, or set --data_file and --image_root."
        )
        return 0

    for i, s in enumerate(samples):
        messages = build_messages(s["image_paths"], s["question"])
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        # Decode only the new tokens
        response_ids = out[0][inputs["input_ids"].shape[1] :]
        answer_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        print(f"--- Sample {i + 1} ---")
        print(f"Images ({len(s['image_paths'])}): {', '.join(s['image_paths'])}")
        print(f"Q: {s['question'][:200]}{'...' if len(s['question']) > 200 else ''}")
        print(f"Ground truth: {s['ground_truth'][:200]}{'...' if len(s['ground_truth']) > 200 else ''}")
        print(f"Model:        {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
