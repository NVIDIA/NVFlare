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
import os
import subprocess
import sys
import tempfile

import nvflare.client as flare

from model import Qwen3VLModel


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL SFT script as subprocess per FL round")
    parser.add_argument("--data_path", type=str, default="./data/site-1", help="Site data dir (train.json here)")
    parser.add_argument("--qwen_root", type=str, default=None, help="Qwen3-VL repo root (or set QWEN3VL_ROOT)")
    parser.add_argument("--dataset_use", type=str, default="fl_site", help="Dataset name for train_qwen.py (must exist in Qwen data_list)")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per FL round for Qwen script")
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

    use_temp = args.work_dir is None
    work_dir = tempfile.mkdtemp(prefix="qwen3vl_fl_") if use_temp else _abs_path(args.work_dir)
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
        model.model.save_pretrained(input_model_dir)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(input_model_dir)

        # Run Qwen3-VL training script (single process; no torchrun)
        cmd = [
            sys.executable,
            "-u",
            train_script,
            "--model_name_or_path", input_model_dir,
            "--output_dir", output_model_dir,
            "--dataset_use", args.dataset_use,
            "--max_steps", str(args.max_steps),
            "--data_flatten", "True",
            "--tune_mm_mlp", "True",
            "--tune_mm_llm", "True",
            "--bf16",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", "2e-7",
            "--save_strategy", "no",
            "--report_to", "none",
        ]
        cwd = finetune_dir
        # Allow Qwen package to be imported
        env["PYTHONPATH"] = finetune_dir + os.pathsep + env.get("PYTHONPATH", "")
        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Qwen SFT script failed: {e}", file=sys.stderr)
            # Send last checkpoint so round does not block; fallback to input model
            from transformers import Qwen2_5_VLForConditionalGeneration
            load_dir = output_model_dir if os.path.isdir(output_model_dir) else input_model_dir
            try:
                model.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(load_dir)
            except Exception:
                pass  # keep current model (received weights)
            output_model = flare.FLModel(
                params=model.cpu().state_dict(),
                metrics={"loss": float("nan")},
                meta={"ERROR": str(e)},
            )
            flare.send(output_model)
            continue

        # Load trained model from output_dir and send state_dict
        from transformers import Qwen2_5_VLForConditionalGeneration
        trained = Qwen2_5_VLForConditionalGeneration.from_pretrained(output_model_dir)
        model.model = trained
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"loss": 0.0},
            meta={"NUM_STEPS_CURRENT_ROUND": args.max_steps},
        )
        print(f"site={client_name}, round={input_model.current_round}, sent updated weights")
        flare.send(output_model)

    if use_temp and os.path.isdir(work_dir):
        import shutil
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
