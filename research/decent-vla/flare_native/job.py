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

"""Compose and export a VLA job for nvflare simulator/job submit."""

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import yaml

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# Names of the files the job ships: the client script goes in each app's
# custom/ folder, the run config (and the files it names) in config/.
CLIENT_SCRIPT = "client.py"
RUN_CONFIG = "run.yaml"
SOURCE_DIR = Path(__file__).resolve().parent / "src"


# The files a run config names that `job.staged_job` packages next to it: the
# lerobot training config and the held-out task spec. In the packaged config
# each is the file's bare name.
def packaged_refs(raw: dict) -> list[tuple[str, dict, str]]:
    """Where a raw run config names its packaged files: (config key, the
    mapping that holds it, its field), for each place the loader accepts."""
    extra = (raw.get("backend") or {}).get("extra") or {}
    data = raw.get("data") or {}
    refs = [("backend.extra.train_config_path", extra, "train_config_path"), ("data.held_out", data, "held_out")]
    if isinstance(data.get("partitioning"), dict):  # the `data.partitioning: {scheme, held_out, ...}` form
        refs.append(("data.partitioning.held_out", data["partitioning"], "held_out"))
    return refs


@contextlib.contextmanager
def staged_job(config_path: str):
    """Stage every file the job ships, by the name it has inside the job, and
    run the block from that folder. Yields (stage folder, config file names).

    The job carries its own run config, the files that config names
    (rewritten to bare names) and the client script,
    so a deployed site needs neither the source tree nor the submitter's
    paths. FedJob resolves relative resource names against the cwd both when
    the job is built and when it is exported, so both happen in here.
    """
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="vla_flare_job_") as stage:
        config_files = [RUN_CONFIG]
        for key, holder, field in packaged_refs(raw):
            value = holder.get(field)
            if not value:
                continue
            # Resolve input files against the exporting process cwd.
            src = Path(cwd, value)
            if not src.is_file():
                raise FileNotFoundError(f"{key}={value!r}: no such file under {cwd}")
            if src.name in (RUN_CONFIG, CLIENT_SCRIPT) or src.name in config_files:
                raise ValueError(f"{key}: {src.name!r} is already a file name in the job")
            shutil.copy(src, Path(stage, src.name))
            holder[field] = src.name
            config_files.append(src.name)
        Path(stage, RUN_CONFIG).write_text(yaml.safe_dump(raw, sort_keys=False))
        shutil.copy(Path(__file__).with_name(CLIENT_SCRIPT), Path(stage, CLIENT_SCRIPT))
        os.chdir(stage)
        try:
            yield Path(stage), config_files
        finally:
            os.chdir(cwd)


class JobSourceFiles:
    """Bundle the native source tree into an app's custom/src folder."""

    def __init__(self, root: Path):
        self.root = root

    def add_to_fed_job(self, job, ctx, **kwargs):
        for path in sorted(self.root.rglob("*.py")):
            relative = path.relative_to(self.root)
            destination = str(Path("src") / relative.parent)
            job.add_file_source(str(path), destination, "custom", ctx)


class JobConfigFiles:
    """FedJob component that puts files in the app's config/ folder.

    Not in custom/: `nvflare job submit` parses and rewrites every .yaml,
    .json and .conf file under custom/, but in config/ only its own
    config_fed_* files.
    """

    def __init__(self, paths: list[str]):
        self.paths = list(paths)

    def add_to_fed_job(self, job, ctx, **kwargs):
        for path in self.paths:
            job.add_file_source(path, None, "config", ctx)


def build_job(cfg, staged: tuple[Path, list[str]]):
    """Wire native FedAvg/FedOpt, a LearnablePersistor and Client API trainers."""
    from nvflare.client.config import ExchangeFormat, TransferType
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    from src.persistor import VLAPersistor
    from src.workflows import VLAFedAdam, VLAFedAvg

    # min_clients is the job's min_sites: FLARE aborts the job once fewer sites
    # are alive, so it follows the round loop's quorum.
    job = FedJob(name=cfg.name or "vla-flare", min_clients=cfg.fl.min_clients or cfg.data.num_clients)
    # The whole native source tree is packaged into each app. Mark it as an
    # app package so FedJob does not also copy a partial import graph.
    job.set_app_packages(["src"])
    stage, config_files = staged
    configs = [str(stage / name) for name in config_files]
    source = JobSourceFiles(SOURCE_DIR)
    job.to_server(source)
    job.to_clients(source)
    job.to_server(VLAPersistor(config_path=RUN_CONFIG), id="persistor")
    workflow = VLAFedAdam if cfg.fl.strategy.name == "fedadam" else VLAFedAvg
    job.to_server(
        workflow(
            persistor_id="persistor",
            num_clients=cfg.data.num_clients,
            num_rounds=cfg.fl.num_rounds,
            ignore_result_error=None,
        )
    )
    job.to_server(JobConfigFiles(configs))
    job.to_clients(JobConfigFiles(configs))
    job.to_clients(
        ScriptRunner(
            # A name inside the job: in-process mode finds it in custom/, and
            # external-process mode runs `python3 -u custom/client.py`.
            script=CLIENT_SCRIPT,
            script_args=f"--config {RUN_CONFIG}",
            launch_external_process=cfg.fl.nvflare.launch_external_process,
            # NUMPY end-to-end + FULL transfer = the Client API applies no
            # framework conversion or diffing; params are exactly the clients'
            # wire arrays (deltas are computed in the Client API training loop).
            framework=FrameworkType.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.FULL,
        )
    )
    return job


def export_job(cfg, config_path: str, job_dir: str | Path = "jobs") -> Path:
    """Export once; use nvflare simulator/job submit to execute the same job."""
    job_dir = Path(job_dir).resolve()
    with staged_job(config_path) as staged:
        build_job(cfg, staged).export_job(str(job_dir))
    return job_dir / cfg.name


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
            "save_every_n_rounds": 1 if args.smoke or policy == "mock" else 10,
            "output_dir": args.output_dir or f"outputs/{name}",
        },
    }
    backend, data = cfg["backend"], cfg["data"]
    backend["optim"] = {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0}
    if policy == "mock":
        cfg["eval"] = {
            "benchmark": {"name": "mock", "suites": ["default"]},
            "n_episodes_per_task": 3,
            "seed": 0,
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
    return cfg


def write_config(args):
    from src.configs.loader import load_run_config

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
    parser.add_argument("--output-dir", help="config/checkpoint folder; default outputs/<experiment name>")
    parser.add_argument("--config", help="export an existing run YAML instead of composing a recipe")
    parser.add_argument("--job-dir", default="jobs", help="export destination; default jobs/<experiment name>")
    return parser


def main():
    parser = define_parser()
    args = parser.parse_args()
    from src.configs.loader import load_run_config

    try:
        path = Path(args.config) if args.config else write_config(args)
        cfg = load_run_config(path)
        exported = export_job(cfg, str(path), args.job_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(exported)


if __name__ == "__main__":
    main()
