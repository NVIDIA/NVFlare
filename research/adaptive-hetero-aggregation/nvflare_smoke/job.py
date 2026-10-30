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

"""Fast CPU-only end-to-end smoke run using the generic aggregator with FedOptRecipe."""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

from adaptive_hetero.nvflare_aggregator import AdaptiveHeterogeneityAggregator, AdaptiveMetaKey  # noqa: E402

from nvflare.app_common.app_constant import DefaultCheckpointFileName  # noqa: E402
from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

NUM_FEATURES = 6
NUM_CLASSES = 3


def _torch_load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"unexpected checkpoint payload at {path}: {type(checkpoint).__name__}")
    return checkpoint


def _final_checkpoint_blend(result_workspace: str) -> tuple[float, Path]:
    """Return the final server-side blend factor persisted by the FedOpt smoke run."""

    if not result_workspace:
        raise RuntimeError("NVFlare smoke run did not return a result workspace")
    root = Path(result_workspace)
    if not root.is_dir():
        raise RuntimeError(f"NVFlare smoke result workspace does not exist: {root}")

    candidates = [
        path
        for path in root.rglob(DefaultCheckpointFileName.GLOBAL_MODEL)
        if "server" in path.parts
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one server {DefaultCheckpointFileName.GLOBAL_MODEL} checkpoint under {root}, "
            f"found {len(candidates)}"
        )

    checkpoint_path = candidates[0]
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    meta = checkpoint.get("meta_props") or {}
    if not isinstance(meta, dict):
        raise RuntimeError(f"checkpoint meta_props must be a dict: {checkpoint_path}")
    blend = meta.get(AdaptiveMetaKey.BLEND_FACTOR)
    if blend is None:
        raise RuntimeError(
            f"final server checkpoint is missing {AdaptiveMetaKey.BLEND_FACTOR!r}: {checkpoint_path}"
        )
    try:
        blend = float(blend)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"invalid persisted adaptive blend factor {blend!r}") from exc
    return blend, checkpoint_path


def main(args):
    torch.manual_seed(args.seed)
    model = torch.nn.Linear(NUM_FEATURES, NUM_CLASSES)
    # Use deliberately permissive gates in this smoke run while preserving the
    # production warm-up, patience, and stable-cohort safeguards. Constructor
    # arguments are public attributes so FedJob serializes these values into the
    # server-side component rather than reconstructing it with defaults.
    aggregator = AdaptiveHeterogeneityAggregator(
        metric_prior_strength=0.0,
        min_weight=0.05,
        max_weight=0.60,
        heterogeneity_threshold=0.0,
        heterogeneity_temperature=0.04,
        heterogeneity_deadband=0.0,
        performance_gap_threshold=0.0,
        performance_gap_deadband=0.0,
        activation_warmup_rounds=3,
        activation_patience=2,
        require_stable_cohort=True,
    )
    client_script = os.path.join(os.path.dirname(__file__), "client.py")
    recipe = FedOptRecipe(
        name="adaptive-hetero-smoke",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script=client_script,
        train_args=(
            f"--train_samples {args.train_samples} --valid_samples {args.valid_samples} "
            f"--local_epochs {args.local_epochs} --batch_size {args.batch_size} --seed {args.seed}"
        ),
        aggregator=aggregator,
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
        device="cpu",
    )

    run = recipe.execute(SimEnv(num_clients=args.n_clients))
    status = str(run.get_status())
    result_workspace = run.get_result()
    print(f"ADAPTIVE_HETERO_SMOKE_STATUS={status}")
    print(f"ADAPTIVE_HETERO_SMOKE_RESULT={result_workspace}")
    if "COMPLETED" not in status.upper():
        raise RuntimeError(f"NVFlare smoke run did not complete successfully: {status}")

    blend_factor, checkpoint_path = _final_checkpoint_blend(result_workspace)
    print(f"ADAPTIVE_HETERO_SMOKE_FINAL_BLEND={blend_factor}")
    print(f"ADAPTIVE_HETERO_SMOKE_CHECKPOINT={checkpoint_path}")
    if blend_factor <= 0.0:
        raise RuntimeError(
            "NVFlare smoke run completed without exercising adaptive weighting; "
            f"final persisted blend factor was {blend_factor}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--train_samples", type=int, default=180)
    parser.add_argument("--valid_samples", type=int, default=90)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    main(parser.parse_args())
