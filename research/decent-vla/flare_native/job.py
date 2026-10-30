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

"""Compose a VLA experiment and run/export its native FLARE job."""

import argparse
from pathlib import Path

import yaml

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_pipeline(policy, tuning):
    cfg = {
        "policy": {"type": policy, "pretrained_path": f"lerobot/{policy}_base", "push_to_hub": False},
        "dataset": {"repo_id": "HuggingFaceVLA/libero", "revision": "main", "video_backend": "pyav"},
    }
    if policy == "pi05":
        cfg["policy"].update(
            compile_model=False,
            gradient_checkpointing=True,
            empty_cameras=1,
            normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        )
        cameras = ["base_0_rgb", "left_wrist_0_rgb"]
    else:
        features = {"observation.state": {"type": "STATE", "shape": [8]}}
        features.update(
            {f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]} for i in (1, 2, 3)}
        )
        cfg["policy"].update(input_features=features, output_features={"action": {"type": "ACTION", "shape": [7]}})
        if tuning == "fullft":
            cfg["policy"].update(train_expert_only=False, freeze_vision_encoder=False)
        cameras = ["camera1", "camera2"]
    cfg["rename_map"] = {
        f"observation.images.{src}": f"observation.images.{dst}" for src, dst in zip(("image", "image2"), cameras)
    }
    return cfg


def create_config(args):
    policy, lora = args.policy, args.tuning == "lora"
    if policy == "mock" and lora:
        raise ValueError("the CPU mock uses full tuning")
    name = "mock_fl" if policy == "mock" else f"{policy}_libero_{args.tuning}"
    name += (f"_{args.algorithm}" if args.algorithm != "fedavg" else "") + "_nvflare"
    if args.smoke:
        name += "_smoke"
    clients = args.num_clients if args.num_clients is not None else 2 if args.smoke or policy == "mock" else 8
    rounds = (
        args.num_rounds
        if args.num_rounds is not None
        else 2 if args.smoke else {"mock": 2, "pi05": 100, "smolvla": 160, "groot": 140}[policy]
    )
    steps = args.local_steps if args.local_steps is not None else 2 if policy == "mock" else 20 if args.smoke else 250
    if min(clients, rounds, steps) < 1:
        raise ValueError("client, round and local-step counts must be positive")
    cfg = {
        "name": name,
        "seed": args.seed,
        "device": "cpu" if policy == "mock" else "auto",
        "backend": {"name": "mock" if policy == "mock" else "groot" if policy == "groot" else "lerobot"},
        "data": {
            "name": "mock" if policy == "mock" else "groot" if policy == "groot" else "lerobot",
            "batch_size": 8,
            "num_workers": 0,
            "num_clients": clients,
            "partitioning": "iid" if policy == "mock" else "task_round_robin",
            "seed": args.seed,
        },
        "fl": {
            "strategy": {"name": args.algorithm},
            "num_rounds": rounds,
            "local_steps": steps,
            "mode": "nvflare",
            "save_every_n_rounds": 1 if args.smoke or policy == "mock" else 10,
            "output_dir": args.output_dir or f"outputs/{name}",
            "nvflare": {"env": "export" if args.export else "simulator", "threads": args.threads or clients},
        },
    }
    backend, data = cfg["backend"], cfg["data"]
    backend["optim"] = {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0}
    if policy == "mock":
        cfg["eval"] = {
            "benchmark": {"name": "mock", "suites": ["default"]},
            "n_episodes_per_task": 3,
            "seed": 0,
            "after_training": True,
        }
    if policy != "mock":
        backend["optim"].update(betas=[0.9, 0.95], scheduler="cosine", warmup_steps=1000, decay_steps=30000)
        backend["extra"] = {"dtype": "bfloat16"}
        if policy == "groot":
            paths = [f"data/groot-libero/{suite}_no_noops_1.0.0_lerobot" for suite in SUITES]
            backend["pretrained_path"] = "nvidia/GR00T-N1.7-3B"
            backend["optim"].update(lr=1e-4, weight_decay=1e-5, decay_lr=1e-5)
            backend["extra"].update(
                embodiment_tag="LIBERO_PANDA",
                state_dropout_prob=0.2,
                gradient_checkpointing=True,
                eval_dataset_paths=paths,
                local_files_only=True,
                tune_llm=False,
                tune_visual=False,
                tune_projector=True,
                tune_diffusion_model=not lora,
                tune_vlln=not lora,
            )
            data.update(
                batch_size=32 if lora else 16,
                num_workers=16,
                extra={
                    "video_backend": "pyav",
                    "dataloader_mp_context": "spawn",
                    "episode_sampling_rate": 1.0,
                    "dataset_paths": paths,
                },
            )
            cfg["eval"] = {
                "benchmark": {"name": "libero", "suites": SUITES[:1] if args.smoke else SUITES},
                "n_episodes_per_task": 1 if args.smoke else 10,
                "eval_all_tasks": True,
                "seed": args.seed,
            }
        else:
            data["repo_id"] = "HuggingFaceVLA/libero"
            if policy == "pi05":
                backend["extra"].update(
                    policy_type="pi05",
                    pretrained_path="lerobot/pi05_base",
                    tokenizer_local_repo="google/paligemma-3b-pt-224",
                )
                backend["optim"].update(
                    lr=2.5e-4 if lora else 2.5e-5, weight_decay=0.01, decay_lr=2.5e-5 if lora else 2.5e-6
                )
                data.update(
                    batch_size=32,
                    num_workers=4 if lora else 2,
                    extra={"dataloader_mp_context": "forkserver", "prefetch_factor": 1},
                )
            else:
                backend["optim"].update(lr=1e-4, eps=1e-8, weight_decay=1e-10, decay_lr=2.5e-5)
                if not lora:
                    backend["optim"]["grad_clip_norm"] = 10
                    data["extra"] = {"dataloader_mp_context": "spawn"}
                data.update(batch_size=64 if lora else 32, num_workers=8 if lora else 12)
        if lora:
            backend["lora"] = {"r": 64, "alpha": 128, "dropout": 0.0, "target_modules": LORA_TARGETS}
            if policy == "groot":
                backend["lora"].update(
                    attach_to="action_head.model",
                    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
                )
    if args.algorithm == "fedprox":
        cfg["fl"]["strategy"].update(mu=0.01, extra={"normalize": policy == "mock"})
    elif args.algorithm == "fedadam":
        cfg["fl"]["strategy"].update(server_lr=0.001, server_betas=[0.9, 0.999], server_eps=0.001)
    if args.batch_size is not None:
        data["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if policy != "mock" or args.gpus:
        ids = (
            args.gpus.split(",")
            if args.gpus
            else [str(i) for i in range(min(clients, 8 if policy == "pi05" and not lora else 4))]
        )
        if any(not value.strip() for value in ids):
            raise ValueError("--gpus must be a comma-separated list of device IDs")
        cfg["fl"]["nvflare"]["gpu"] = ",".join(ids[i % len(ids)].strip() for i in range(clients))
    return cfg


def write_config(args):
    from decent_vla.configs import load_run_config

    cfg = create_config(args)
    folder = Path(cfg["fl"]["output_dir"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.policy in ("pi05", "smolvla"):
        pipeline = folder / "train_pipeline.yaml"
        pipeline.write_text(yaml.safe_dump(train_pipeline(args.policy, args.tuning), sort_keys=False))
        cfg["backend"]["extra"]["train_config_path"] = str(pipeline)
    path = folder / "run.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    load_run_config(path)
    return path


def define_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["mock", "pi05", "smolvla", "groot"], default="mock")
    parser.add_argument("--tuning", choices=["fullft", "lora"], default="fullft")
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox", "fedadam"], default="fedavg")
    parser.add_argument("--smoke", action="store_true", help="two clients, two rounds, 20 local steps for VLA policies")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", help="available device IDs, e.g. 0,1; expands to one resident slot per client")
    parser.add_argument("--threads", type=int, help="simulator threads; defaults to client count")
    parser.add_argument("--output-dir", help="config/checkpoint folder; default outputs/<experiment name>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="write run/pipeline YAML without starting FLARE")
    mode.add_argument("--export", action="store_true", help="export a deployable FLARE job instead of simulating")
    return parser


def main():
    parser = define_parser()
    args = parser.parse_args()
    try:
        path = write_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(path, flush=True)
    if not args.generate_only:
        from decent_vla.cli.main import main as run

        run(["train", "--config", str(path)])


if __name__ == "__main__":
    main()
