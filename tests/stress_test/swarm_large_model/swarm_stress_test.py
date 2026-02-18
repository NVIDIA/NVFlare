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

"""Swarm learning stress test with large PyTorch models across 3 sites.

Exercises the DXO aggregation path (CCWF/Swarm → DXOAggregator →
WeightedAggregationHelper) with PyTorch tensors to validate disk-streamed
tensor aggregation.

Usage:
    # Smoke test (100 MB model)
    python swarm_stress_test.py --model-size-gb 0.1 --num-rounds 1

    # Full stress test (5 GB model)
    python swarm_stress_test.py --model-size-gb 5 --num-rounds 2

    # Disable disk streaming (baseline memory comparison)
    python swarm_stress_test.py --model-size-gb 1 --no-disk-streaming

    # Compare outputs (run both modes, verify identical results)
    python swarm_stress_test.py --model-size-gb 0.1 --num-rounds 1 --compare
"""

import argparse
import glob
import hashlib
import os
import shutil
import sys

import torch

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs
from nvflare.security.logging import secure_format_exception

NUM_LAYERS = 50
RESULT_FILENAME = "stress_test_final_model.pt"


def _build_state_dict(size_gb: float) -> dict[str, torch.Tensor]:
    total_elements = int(size_gb * (1024**3) / 4)
    per_layer = total_elements // NUM_LAYERS
    remainder = total_elements - per_layer * NUM_LAYERS

    state_dict = {}
    for i in range(NUM_LAYERS):
        n = per_layer + (1 if i < remainder else 0)
        state_dict[f"layer_{i}.weight"] = torch.ones(n, dtype=torch.float32)

    actual_gb = sum(t.nelement() * 4 for t in state_dict.values()) / (1024**3)
    print(f"  State dict: {len(state_dict)} tensors, {total_elements:,} elements, {actual_gb:.2f} GB")
    return state_dict


def _checksum_state_dict(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class LargePTModelPersistor(ModelPersistor):
    def __init__(self, size_gb: float):
        super().__init__()
        self.size_gb = size_gb

    def load_model(self, fl_ctx: FLContext):
        fobs.register(TensorDecomposer)
        self.log_info(fl_ctx, f"Creating initial PT model (~{self.size_gb:.1f} GB)")
        state_dict = _build_state_dict(self.size_gb)
        return make_model_learnable(weights=state_dict, meta_props={})

    def save_model(self, model_learnable, fl_ctx: FLContext):
        weights = model_learnable.get(ModelLearnableKey.WEIGHTS, {})
        if not weights:
            self.log_info(fl_ctx, "No weights to save")
            return

        # Resolve any lazy refs before saving
        for k, v in list(weights.items()):
            if hasattr(v, "resolve"):
                weights[k] = v.resolve()

        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        result_dir = engine.get_workspace().get_run_dir(job_id)
        save_path = os.path.join(result_dir, RESULT_FILENAME)

        checksum = _checksum_state_dict(weights)
        self.log_info(fl_ctx, f"Saving final model: {len(weights)} keys, checksum={checksum}")
        torch.save(weights, save_path)
        self.log_info(fl_ctx, f"Saved to {save_path}")


class LargePTTrainer(Executor):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self._delta = delta
        fobs.register(TensorDecomposer)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != AppConstants.TASK_TRAIN:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        try:
            dxo = from_shareable(shareable)
        except Exception as e:
            self.system_panic(f"Cannot convert shareable: {secure_format_exception(e)}", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Expected DataKind.WEIGHTS", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        # Resolve lazy refs from disk-streamed download
        for k, v in list(dxo.data.items()):
            if hasattr(v, "resolve"):
                dxo.data[k] = v.resolve()

        total_elements = sum(t.nelement() for t in dxo.data.values() if isinstance(t, torch.Tensor))
        total_gb = total_elements * 4 / (1024**3)
        self.log_info(
            fl_ctx,
            f"Round {current_round}/{total_rounds} – "
            f"{len(dxo.data)} tensors, {total_elements:,} elements ({total_gb:.1f} GB)",
        )

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        for t in dxo.data.values():
            if isinstance(t, torch.Tensor):
                t.add_(self._delta)

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=dxo.data,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        return outgoing_dxo.to_shareable()


def build_job(model_size_gb: float, num_rounds: int) -> CCWFJob:
    job = CCWFJob(name="swarm_pt_stress_test", min_clients=3)

    job.add_swarm(
        server_config=SwarmServerConfig(
            num_rounds=num_rounds,
            start_task_timeout=3600,
            progress_timeout=600.0,
        ),
        client_config=SwarmClientConfig(
            executor=LargePTTrainer(delta=1.0),
            aggregator=InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS),
            persistor=LargePTModelPersistor(size_gb=model_size_gb),
            shareable_generator=SimpleModelShareableGenerator(),
            learn_task_ack_timeout=3600,
            final_result_ack_timeout=3600,
            min_responses_required=3,
            wait_time_after_min_resps_received=30.0,
        ),
    )
    return job


def _find_result_file(workdir: str) -> str | None:
    pattern = os.path.join(workdir, "**", RESULT_FILENAME)
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def run_one(model_size_gb: float, num_rounds: int, workdir: str, disk_streaming: bool):
    if disk_streaming:
        os.environ["NVFLARE_TENSOR_DOWNLOAD_TO_DISK"] = "true"
    else:
        os.environ.pop("NVFLARE_TENSOR_DOWNLOAD_TO_DISK", None)

    mode = "DISK" if disk_streaming else "MEMORY"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Model: {model_size_gb} GB, Rounds: {num_rounds}")
    print(f"{'='*60}\n")

    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    job = build_job(model_size_gb=model_size_gb, num_rounds=num_rounds)
    job.simulator_run(workdir, n_clients=3)

    result_file = _find_result_file(workdir)
    if result_file:
        sd = torch.load(result_file, weights_only=True)
        checksum = _checksum_state_dict(sd)
        print(f"\n  Result: {result_file}")
        print(f"  Keys: {len(sd)}, Checksum: {checksum}")
        return result_file
    else:
        print(f"\n  WARNING: no result file found in {workdir}")
        return None


def compare_mode(args):
    workdir_disk = args.workdir + "_disk"
    workdir_mem = args.workdir + "_mem"

    f_disk = run_one(args.model_size_gb, args.num_rounds, workdir_disk, disk_streaming=True)
    f_mem = run_one(args.model_size_gb, args.num_rounds, workdir_mem, disk_streaming=False)

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    if not f_disk:
        print("  ERROR: disk mode result not found")
        return False
    if not f_mem:
        print("  ERROR: memory mode result not found")
        return False

    sd_disk = torch.load(f_disk, weights_only=True)
    sd_mem = torch.load(f_mem, weights_only=True)

    if set(sd_disk.keys()) != set(sd_mem.keys()):
        print(f"  FAIL: different keys")
        return False

    max_diff = 0.0
    for k in sorted(sd_disk.keys()):
        diff = (sd_disk[k] - sd_mem[k]).abs().max().item()
        max_diff = max(max_diff, diff)

    cksum_disk = _checksum_state_dict(sd_disk)
    cksum_mem = _checksum_state_dict(sd_mem)

    print(f"  Disk checksum:   {cksum_disk}")
    print(f"  Memory checksum: {cksum_mem}")
    print(f"  Max diff:        {max_diff}")

    if cksum_disk == cksum_mem:
        print(f"  RESULT: IDENTICAL")
        return True
    elif max_diff < 1e-6:
        print(f"  RESULT: NUMERICALLY EQUIVALENT (max diff {max_diff})")
        return True
    else:
        print(f"  RESULT: MISMATCH")
        return False


def main():
    parser = argparse.ArgumentParser(description="Swarm stress test with large PyTorch models")
    parser.add_argument("--model-size-gb", type=float, default=5.0, help="Model size in GB (default: 5)")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of swarm rounds (default: 2)")
    parser.add_argument(
        "--workdir", type=str, default="/tmp/nvflare/swarm_pt_stress", help="Simulator working directory"
    )
    parser.add_argument(
        "--no-disk-streaming", action="store_true", help="Disable disk-based tensor streaming (baseline)"
    )
    parser.add_argument("--compare", action="store_true", help="Run both modes and compare results")
    args = parser.parse_args()

    if args.compare:
        ok = compare_mode(args)
        sys.exit(0 if ok else 1)

    disk_streaming = not args.no_disk_streaming
    run_one(args.model_size_gb, args.num_rounds, args.workdir, disk_streaming)


if __name__ == "__main__":
    main()
