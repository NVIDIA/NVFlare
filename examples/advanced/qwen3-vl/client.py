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
Client-side training script for federated Qwen2.5-VL SFT.
Loads site-specific PubMedVision shard, receives global model, trains one round, sends update.
"""
import argparse
import os
from io import BytesIO

import torch
from torch.utils.data import DataLoader, Dataset

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from model import Qwen2VLModelWrapper


def _load_image(entry, data_path: str, image_root: str = None):
    """Load image from data entry: either bytes or path relative to image_root/data_path."""
    if entry is None:
        return None
    if isinstance(entry, bytes):
        from PIL import Image
        return Image.open(BytesIO(entry)).convert("RGB")
    if isinstance(entry, dict) and "bytes" in entry:
        from PIL import Image
        return Image.open(BytesIO(entry["bytes"])).convert("RGB")
    if isinstance(entry, (list, tuple)) and len(entry) > 0:
        return _load_image(entry[0], data_path, image_root)
    if isinstance(entry, str):
        root = image_root if image_root else data_path
        path = os.path.join(root, entry)
        if os.path.isfile(path):
            from PIL import Image
            return Image.open(path).convert("RGB")
    return None


class PubMedVisionDataset(Dataset):
    """Simple dataset over JSON rows: conversations + images for Qwen2.5-VL."""

    def __init__(self, json_path, processor, data_path: str, image_root: str = None, max_samples: int = None):
        from datasets import load_dataset
        self.data_path = data_path
        self.image_root = image_root
        self.processor = processor
        ds = load_dataset("json", data_files=json_path, split="train")
        if max_samples is not None and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        self.ds = ds
        self.has_image = "image" in ds.column_names
        self.has_conversations = "conversations" in ds.column_names

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        images = []
        if self.has_image and row.get("image"):
            img = _load_image(row["image"], self.data_path, self.image_root)
            if img is not None:
                images.append(img)
        if not images:
            # Fallback: return a dummy so we can still build a batch (processor may need at least one)
            from PIL import Image
            images = [Image.new("RGB", (224, 224), color="gray")]
        conv = row.get("conversations", [])
        if not conv and "conversation" in row:
            conv = row["conversation"]
        if not conv:
            conv = [{"from": "human", "value": "Describe the image."}, {"from": "gpt", "value": "No caption."}]
        human_val = conv[0].get("value", conv[0].get("content", "")) if conv else ""
        gpt_val = conv[1].get("value", conv[1].get("content", "")) if len(conv) > 1 else ""

        # Qwen2.5-VL: user content = [image, text], assistant content = text
        user_content = []
        if images:
            user_content.append({"type": "image", "image": images[0]})
        user_content.append({"type": "text", "text": human_val})
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gpt_val},
        ]
        # Processor may return pixel_values when messages include image content
        try:
            inputs = self.processor(
                messages,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            )
        except Exception:
            # Fallback: tokenize text only, then get pixel_values from image_processor
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            )
            if images and hasattr(self.processor, "image_processor"):
                inputs["pixel_values"] = self.processor.image_processor(images, return_tensors="pt")["pixel_values"]
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch):
    """Collate batch; stack pixel_values when present."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    pixel_values = None
    pv_list = [b["pixel_values"] for b in batch if b.get("pixel_values") is not None]
    if pv_list:
        pixel_values = torch.cat(pv_list, dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/site-1", help="Path to site data (dir with train.json)")
    parser.add_argument("--image_root", type=str, default=None, help="Root for image paths in JSON (e.g. PubMedVision repo)")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps_per_round", type=int, default=50, help="Cap steps per FL round for quick runs")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples from JSON (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    json_path = os.path.join(args.data_path, "train.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Expected train.json at {json_path}. Run prepare_data.py first.")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2VLModelWrapper(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else "cpu",
    )
    model.to(device)
    model.train()

    dataset = PubMedVisionDataset(
        json_path,
        processor,
        args.data_path,
        image_root=args.image_root,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    summary_writer = SummaryWriter()

    flare.init()
    client_name = flare.system_info().get("site_name", "unknown")

    while flare.is_running():
        input_model = flare.receive()
        print(f"site={client_name}, round={input_model.current_round}")

        # Load global weights
        model.load_state_dict(input_model.params, strict=False)
        model.to(device)

        if flare.is_evaluate():
            # Optional: compute eval metric and return
            output_model = flare.FLModel(metrics={"loss": 0.0})
            flare.send(output_model)
            continue

        steps = 0
        total_loss = 0.0
        for epoch in range(args.epochs):
            for batch in dataloader:
                if args.max_steps_per_round is not None and steps >= args.max_steps_per_round:
                    break
                for k, v in batch.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype)
                optimizer.zero_grad()
                forward_kw = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }
                if batch.get("pixel_values") is not None:
                    forward_kw["pixel_values"] = batch["pixel_values"]
                outputs = model.model(**forward_kw)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                if steps % 10 == 0:
                    summary_writer.add_scalar("loss", loss.item(), input_model.current_round * 1000 + steps)
            if args.max_steps_per_round is not None and steps >= args.max_steps_per_round:
                break

        avg_loss = total_loss / max(steps, 1)
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"loss": avg_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site={client_name}, round={input_model.current_round}, steps={steps}, loss={avg_loss:.4f}")
        flare.send(output_model)


if __name__ == "__main__":
    main()
