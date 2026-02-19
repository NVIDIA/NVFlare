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
Client script that runs the Qwen3-VL fine-tuning script as an external process per FL round.
Receives global model from NVFlare, saves to a dir, invokes train_qwen.py (from Qwen3-VL repo),
loads the saved checkpoint, and sends updated weights back.
Requires QWEN3VL_ROOT and (for site data) a "fl_site" dataset entry in the Qwen repo's data_list
that reads FL_SITE_DATA_DIR (see README).
"""
import argparse
import gc
import os
import subprocess
import sys
import tempfile

import nvflare.client as flare
import torch

from model import Qwen3VLModel, load_qwen_vl_from_pretrained, load_state_dict_from_checkpoint


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


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

    # So that Qwen data_list can resolve "fl_site" from FL_SITE_DATA_DIR
    env = os.environ.copy()
    env["FL_SITE_DATA_DIR"] = data_path

    model = Qwen3VLModel(model_name_or_path=args.model_name_or_path)

    while flare.is_running():
        input_model = flare.receive()
        print(f"site={client_name}, round={input_model.current_round}")

        if flare.is_evaluate():
            output_model = flare.FLModel(metrics={"loss": 0.0})
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

        # Run Qwen3-VL training via wrapper so we can call destroy_process_group() on exit (avoids PyTorch warning)
        wrapper_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_train_with_cleanup.py")
        env["QWEN_FINETUNE_DIR"] = finetune_dir
        # Unique master port per client to avoid TCPStore port conflict when multiple clients run on same host (simulator)
        try:
            site_num = int(client_name.split("-")[-1])
        except (ValueError, IndexError):
            site_num = 0
        master_port = 29500 + (site_num if site_num > 0 else 0)
        train_limit = (
            ["--max_steps", str(args.max_steps)]
            if args.max_steps is not None
            else ["--num_train_epochs", str(args.num_train_epochs)]
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--nnodes=1",
            "--master_port",
            str(master_port),
            wrapper_script,
            "--model_name_or_path", input_model_dir,
            "--output_dir", output_model_dir,
            "--dataset_use", args.dataset_use,
            *train_limit,
            "--data_flatten", "True",
            "--tune_mm_mlp", "True",
            "--tune_mm_llm", "True",
            "--bf16",
            "--per_device_train_batch_size", "8",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", args.learning_rate,
            "--save_strategy", "no",
            "--report_to", "none",
            "--ddp_find_unused_parameters", "False",
        ]
        cwd = finetune_dir
        # Allow Qwen package to be imported
        env["PYTHONPATH"] = finetune_dir + os.pathsep + env.get("PYTHONPATH", "")
        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as e:
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
        print(f"site={client_name}, round={input_model.current_round}, sent updated weights")
        flare.send(output_model)

        # Free memory before next round to reduce OOM risk in long runs (e.g. round 5+)
        del raw, params, output_model
        _free_memory_after_send()

    # When using default work_dir, checkpoints live under the NVFlare client workspace; SimEnv clears it every run.


if __name__ == "__main__":
    main()
