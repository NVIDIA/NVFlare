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

"""FedRevive policy attached to either actual-arrival or research async training."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from fedrevive import (
    ClassProportionSource,
    SynthesisLabelDistribution,
    TeacherBuffer,
    blend_updates,
    cosine_staleness_beta,
)
from model import set_seed

from nvflare.collab import collab


class FedAvgUpdate:
    """Retain the reference FedAvg same-base requirement for each buffer."""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self._versions = []

    def __call__(self, update):
        self._versions.append(update.assignment.base_version)
        if len(self._versions) == self.buffer_size:
            if len(set(self._versions)) != 1:
                raise RuntimeError("FedAvg buffer contains updates from different global versions")
            self._versions.clear()
        return update.delta


class FedReviveUpdate:
    """Collect the arriving teacher, then optionally revive its exact-base delta.

    The async coordinator supplies client deltas relative to assignment bases.
    The reviver returns KD deltas relative to current_model. This policy blends
    those deltas before the coordinator buffers them. No scheduler is imported.
    Full model uploads are required to retain teacher feature statistics.
    """

    def __init__(
        self,
        prepared_data_root,
        device=None,
        generation_interval=10,
        class_proportion_source="estimated",
        probe_count=64,
        run_seed=10,
        synthesis_label_distribution=SynthesisLabelDistribution.UNIFORM.value,
    ):
        self.prepared_data_root = prepared_data_root
        self.device = device
        self.generation_interval = generation_interval
        self.class_proportion_source = class_proportion_source
        self.probe_count = probe_count
        self.run_seed = run_seed
        self.synthesis_label_distribution = synthesis_label_distribution

    def initialize(self):
        from data import load_manifest
        from dfkd import ClassProportionProxyEstimator, DFKDConfig, DFKDReviver

        self.config = DFKDConfig(
            generation_interval=self.generation_interval,
            proxy_batch_size=self.probe_count,
            synthesis_label_distribution=self.synthesis_label_distribution,
        )
        self.teachers = TeacherBuffer(self.config.teacher_buffer_size)
        self.run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        (self.run_dir / "dfkd_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reviver = DFKDReviver(device, output_dir=str(self.run_dir / "dfkd"), config=self.config)
        self.estimator = None
        self.proportions = None
        if ClassProportionSource(self.class_proportion_source) is ClassProportionSource.ESTIMATED:
            self.estimator = ClassProportionProxyEstimator(
                device,
                num_uploads=self.config.proxy_num_uploads,
                temperature=self.config.proxy_temperature,
                batch_size=self.config.proxy_batch_size,
                seed=self.run_seed,
            )
        else:
            self.proportions = load_manifest(self.prepared_data_root)["class_proportions"]
        # Start the server's run stream after allocating every DFKD model.
        # Their initialization depends on the setup seed, not the run seed.
        set_seed(self.run_seed)

    def __call__(self, update):
        from data import prepared_shard_name

        if update.metadata.get("is_delta", False):
            raise ValueError("FedRevive requires full model uploads for its teacher history")
        name = prepared_shard_name(update.assignment.logical_name)
        returned = update.returned_model
        proxy = (
            self.estimator.estimate(name, returned)
            if self.estimator is not None
            else torch.tensor(self.proportions[name], dtype=torch.float32, device="cpu")
        )
        self.teachers.add(returned, proxy, name, update.assignment.base_version)
        beta = cosine_staleness_beta(update.staleness, 75)
        eligible = update.current_version >= self.config.freeze_versions and beta > 0
        distilled, metrics = None, {}
        if eligible:
            distilled, metrics = self.reviver.revive(update.current_model, list(self.teachers), update.current_version)
        with (self.run_dir / "revival.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "assignment_id": update.assignment.assignment_id,
                        "current_version": update.current_version,
                        "staleness": update.staleness,
                        "beta": beta,
                        "revival_applied": distilled is not None,
                        **metrics,
                    }
                )
                + "\n"
            )
        if distilled is not None:
            return blend_updates(update.delta, distilled, beta)
        if eligible:
            # During synthesis-only warmup the student delta is zero, so the
            # hybrid update still applies the stale client weight from Eq. (3).
            return {name: (1.0 - beta) * value for name, value in update.delta.items()}
        return update.delta

    def close(self):
        # The workflow owns shutdown; release persistent models on success/failure.
        self.reviver = self.estimator = self.teachers = self.proportions = None
