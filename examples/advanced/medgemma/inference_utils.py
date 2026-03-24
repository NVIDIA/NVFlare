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

from __future__ import annotations

import os

import torch
from data_utils import DEFAULT_MODEL_NAME_OR_PATH, PROMPT, TISSUE_CLASSES, parse_prediction_label
from model import apply_adapter_state, create_peft_medgemma_model, load_medgemma_base_model
from peft import PeftModel
from transformers import AutoProcessor

NVFLARE_PT_MODEL_KEY = "model"


def load_nvflare_global_pt(pt_path: str) -> dict:
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {pt_path}, got {type(data)}")
    state_dict = data.get(NVFLARE_PT_MODEL_KEY)
    if state_dict is None:
        raise ValueError(f"No key {NVFLARE_PT_MODEL_KEY!r} in {pt_path}. Keys: {list(data.keys())}")
    return state_dict


def load_model_and_processor(
    model_path: str,
    base_model: str = DEFAULT_MODEL_NAME_OR_PATH,
    device: str = "cuda",
):
    quantized = device.startswith("cuda")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    processor.tokenizer.padding_side = "left"

    if os.path.isfile(model_path) and model_path.endswith(".pt"):
        print(f"Loading NVFlare global adapter weights from: {model_path}")
        model = create_peft_medgemma_model(base_model, quantized=quantized, device_map={"": 0} if quantized else None)
        apply_adapter_state(model, load_nvflare_global_pt(model_path))
    elif os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "adapter_config.json")):
        print(f"Loading base model + adapter directory from: {model_path}")
        base = load_medgemma_base_model(base_model, quantized=quantized, device_map={"": 0} if quantized else None)
        model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    else:
        print(f"Loading model directly from: {model_path}")
        model = load_medgemma_base_model(
            model_name_or_path=model_path,
            quantized=quantized,
            device_map={"": 0} if quantized else None,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        processor.tokenizer.padding_side = "left"

    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    return model, processor


def get_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_response_text(model, processor, image, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(get_model_device(model))
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def predict_label(model, processor, image, max_new_tokens: int) -> tuple[str, int, str]:
    response_text = generate_response_text(model, processor, image, max_new_tokens)
    predicted_index = parse_prediction_label(response_text)
    predicted_label = TISSUE_CLASSES[predicted_index] if predicted_index >= 0 else "<unparsed>"
    return response_text, predicted_index, predicted_label
