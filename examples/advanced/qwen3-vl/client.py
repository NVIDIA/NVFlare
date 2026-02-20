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
Client script for federated Qwen3-VL fine-tuning. Launched by NVFlare as an external process
(via torchrun from the job command). Receives global model, saves to a dir, runs Qwen train_qwen
in-process, then loads the checkpoint and sends updated weights back.
Requires QWEN3VL_ROOT and (for site data) a "fl_site" dataset entry in the Qwen repo's data_list
that reads FL_SITE_DATA_DIR (see README).
"""
import argparse
import gc
import os
import signal
import sys
from typing import Optional

import nvflare.client as flare
import torch

from model import Qwen3VLModel, load_qwen_vl_from_pretrained, load_state_dict_from_checkpoint


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _params_size_mb(params) -> float:
    """Return size of params dict (state_dict) in MB. Handles torch tensors and numpy arrays."""
    if not params:
        return 0.0
    nbytes = 0
    for v in params.values():
        if isinstance(v, torch.Tensor):
            nbytes += v.numel() * v.element_size()
        elif hasattr(v, "nbytes"):  # numpy array
            nbytes += v.nbytes
        else:
            nbytes += 0
    return nbytes / (1024.0 * 1024.0)


def _free_memory_after_send() -> None:
    """Run GC and clear CUDA cache to reduce OOM risk before the next round."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _align_model_config_to_tokenizer(hf_model, tokenizer) -> None:
    """Align model config and generation_config with tokenizer PAD/EOS/BOS so loading does not warn."""
    cfg = hf_model.config
    for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
        tid = getattr(tokenizer, key, None)
        if tid is not None and hasattr(cfg, key):
            setattr(cfg, key, tid)
    if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
        gcfg = hf_model.generation_config
        for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
            tid = getattr(tokenizer, key, None)
            if tid is not None and hasattr(gcfg, key):
                setattr(gcfg, key, tid)


def train(
    finetune_dir: str,
    input_model_dir: str,
    output_model_dir: str,
    dataset_use: str,
    max_steps: Optional[int],
    num_train_epochs: int,
    learning_rate: str,
) -> None:
    """Run Qwen3-VL train_qwen.train() in-process; tear down process group on exit."""
    # train_qwen.train() only reads from sys.argv via HfArgumentParser.parse_args_into_dataclasses()
    # and does not accept training args as parameters, so we set sys.argv before calling it.
    # Ensure Qwen finetune package is importable (train_qwen uses "from trainer import ...")
    finetune_dir = os.path.abspath(finetune_dir)
    if finetune_dir not in sys.path:
        sys.path.insert(0, finetune_dir)
    train_dir = os.path.join(finetune_dir, "qwenvl", "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    train_limit = (
        ["--max_steps", str(max_steps)] if max_steps is not None else ["--num_train_epochs", str(num_train_epochs)]
    )
    argv = (
        ["train_qwen.py", "--model_name_or_path", input_model_dir, "--output_dir", output_model_dir]
        + ["--dataset_use", dataset_use]
        + train_limit
        + [
            "--data_flatten", "True",
            "--tune_mm_mlp", "True",
            "--tune_mm_llm", "True",
            "--bf16",
            "--per_device_train_batch_size", "8",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", learning_rate,
            "--save_strategy", "no",
            "--report_to", "wandb",
            "--ddp_find_unused_parameters", "False",
        ]
    )
    old_argv = sys.argv
    sys.argv = argv
    try:
        from qwenvl.train import train_qwen
        train_qwen.train(attn_implementation="flash_attention_2")
    finally:
        sys.argv = old_argv
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL SFT script as subprocess per FL round")
    parser.add_argument("--data_path", type=str, default="./data/site-1", help="Site data dir (train.json here)")
    parser.add_argument("--qwen_root", type=str, default=None, help="Qwen3-VL repo root (or set QWEN3VL_ROOT)")
    parser.add_argument("--dataset_use", type=str, default="fl_site", help="Dataset name for train_qwen.py (must exist in Qwen data_list)")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID (must be Qwen3-VL to match train_qwen.py)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per FL round (omit to train one epoch per round)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Train epochs per round when --max_steps is not set (default 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=str,
        default="5e-7",
        help="Peak learning rate for Qwen script (default 5e-7; use 2e-7 for more stable, slower convergence)",
    )
    parser.add_argument("--work_dir", type=str, default=None, help="Work dir for input/output models (default: temp)")
    args = parser.parse_args()

    # Exit cleanly on SIGTERM (e.g. when NVFlare stops the job) so torchrun does not log SignalException
    def _sigterm_handler(_signum, _frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    qwen_root = args.qwen_root or os.environ.get("QWEN3VL_ROOT", "")
    if not qwen_root or not os.path.isdir(qwen_root):
        print("Set QWEN3VL_ROOT or pass --qwen_root to the Qwen3-VL repo.", file=sys.stderr)
        sys.exit(1)
    qwen_root = _abs_path(qwen_root)
    finetune_dir = os.path.join(qwen_root, "qwen-vl-finetune")
    train_script = os.path.join(finetune_dir, "qwenvl", "train", "train_qwen.py")
    if not os.path.isfile(train_script):
        print(f"Train script not found: {train_script}", file=sys.stderr)
        sys.exit(1)

    data_path = _abs_path(args.data_path)
    json_path = os.path.join(data_path, "train.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Expected train.json at {json_path}. Run prepare_data.py first.")

    flare.init()
    client_name = flare.system_info().get("site_name", "unknown")

    # When --work_dir is not set, use a subdir of cwd (NVFlare client workspace); SimEnv clears it every run
    if args.work_dir is None:
        work_dir = os.path.join(os.getcwd(), "qwen3vl_checkpoints")
    else:
        work_dir = _abs_path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    input_model_dir = os.path.join(work_dir, "input_model")
    output_model_dir = os.path.join(work_dir, "output")

    # So that Qwen data_list and imports can resolve paths (in-process training reads os.environ)
    os.environ["FL_SITE_DATA_DIR"] = data_path
    os.environ["QWEN_FINETUNE_DIR"] = finetune_dir

    model = Qwen3VLModel(model_name_or_path=args.model_name_or_path)

    while flare.is_running():
        input_model = flare.receive()
        received_mb = _params_size_mb(input_model.params)
        print(f"site={client_name}, round={input_model.current_round}, received model size: {received_mb:.2f} MB")

        if flare.is_evaluate():
            output_model = flare.FLModel(metrics={"loss": 0.0})
            sent_mb = _params_size_mb(getattr(output_model, "params", None) or {})
            print(f"site={client_name}, round={input_model.current_round}, sent model size: {sent_mb:.2f} MB")
            flare.send(output_model)
            continue

        # Save received global model to HF format for train_qwen.py
        os.makedirs(input_model_dir, exist_ok=True)
        model.load_state_dict(input_model.params, strict=False)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        _align_model_config_to_tokenizer(model.model, tokenizer)
        model.model.save_pretrained(input_model_dir)
        processor.save_pretrained(input_model_dir)

        # Run Qwen3-VL training in-process (we are already the torchrun process from the job command)
        try:
            train(
                finetune_dir=finetune_dir,
                input_model_dir=input_model_dir,
                output_model_dir=output_model_dir,
                dataset_use=args.dataset_use,
                max_steps=args.max_steps,
                num_train_epochs=args.num_train_epochs,
                learning_rate=args.learning_rate,
            )
        except Exception as e:
            print(f"Qwen SFT script failed: {e}", file=sys.stderr)
            # Send last checkpoint so round does not block; fallback to input model state_dict
            load_dir = output_model_dir if os.path.isdir(output_model_dir) else input_model_dir
            try:
                raw = load_state_dict_from_checkpoint(load_dir)
                params = {"model." + k: v for k, v in raw.items()}
            except Exception:
                try:
                    model.model = load_qwen_vl_from_pretrained(load_dir)
                    params = model.cpu().state_dict()
                except Exception:
                    params = model.cpu().state_dict()  # keep current model (received weights)
            output_model = flare.FLModel(
                params=params,
                metrics={"loss": float("nan")},
                meta={"ERROR": str(e)},
            )
            sent_mb = _params_size_mb(params)
            print(f"site={client_name}, round={input_model.current_round}, sent model size: {sent_mb:.2f} MB (after error)")
            flare.send(output_model)
            del params, output_model
            _free_memory_after_send()
            continue

        # Load state_dict from checkpoint dir (no full model load) so we return to receive() quickly.
        # Checkpoint has inner model keys; prefix with "model." to match wrapper state_dict format.
        raw = load_state_dict_from_checkpoint(output_model_dir)
        params = {"model." + k: v for k, v in raw.items()}
        meta = (
            {"NUM_STEPS_CURRENT_ROUND": args.max_steps}
            if args.max_steps is not None
            else {"NUM_TRAIN_EPOCHS_CURRENT_ROUND": args.num_train_epochs}
        )
        output_model = flare.FLModel(
            params=params,
            metrics={"loss": 0.0},
            meta=meta,
        )
        sent_mb = _params_size_mb(params)
        print(f"site={client_name}, round={input_model.current_round}, sent updated weights, model size: {sent_mb:.2f} MB")
        flare.send(output_model)

        # Free memory before next round to reduce OOM risk in long runs (e.g. round 5+)
        del raw, params, output_model
        _free_memory_after_send()

    # When using default work_dir, checkpoints live under the NVFlare client workspace; SimEnv clears it every run.


if __name__ == "__main__":
    main()
