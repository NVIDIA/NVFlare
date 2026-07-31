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

"""Validate the Swarm tensor disk-offload integration job."""

import json
import os

import torch

from nvflare.app_common.app_constant import DefaultCheckpointFileName

from .job_result_validator import FinishJobResultValidator

OFFLOAD_OBSERVATIONS_FILE = "tensor_offload_observations.jsonl"


class SwarmTensorOffloadValidator(FinishJobResultValidator):
    def __init__(self, expected_value: float = 3.0):
        super().__init__()
        self.expected_value = expected_value

    def validate_finished_results(self, job_result, client_props) -> bool:
        expected_rounds = {"site-1": 0, "site-2": 1}
        job_id = job_result["job_id"]

        for client_prop in client_props:
            expected_round = expected_rounds.get(client_prop.name)
            if expected_round is None:
                self.logger.error(f"unexpected client in tensor offload integration test: {client_prop.name}")
                return False

            run_dir = os.path.join(client_prop.root_dir, job_id)
            if not self._validate_observations(run_dir, client_prop.name, expected_round):
                return False
            if not self._validate_checkpoint(run_dir, client_prop.name):
                return False

        return True

    def _validate_observations(self, run_dir: str, client_name: str, expected_round: int) -> bool:
        marker_path = os.path.join(run_dir, OFFLOAD_OBSERVATIONS_FILE)
        if not os.path.isfile(marker_path):
            self.logger.error(f"offload observation file is missing for {client_name}: {marker_path}")
            return False

        try:
            with open(marker_path, "r", encoding="utf-8") as marker:
                observations = [json.loads(line) for line in marker if line.strip()]
        except Exception as e:
            self.logger.error(f"failed to read offload observations for {client_name}: {e}")
            return False

        if len(observations) != 2:
            self.logger.error(f"expected 2 contributions on {client_name}, got {len(observations)}")
            return False

        for observation in observations:
            if observation.get("site") != client_name or observation.get("round") != expected_round:
                self.logger.error(f"unexpected offload observation on {client_name}: {observation}")
                return False
            if observation.get("total_bytes", 0) <= 0 or not observation.get("file_paths"):
                self.logger.error(f"offload observation has no tensor data on {client_name}: {observation}")
                return False
            for path in observation.get("file_paths", []):
                if os.path.exists(path):
                    self.logger.error(f"offload tensor file was not cleaned after END_RUN: {path}")
                    return False
            for root_dir in observation.get("root_dirs", []):
                if os.path.exists(root_dir):
                    self.logger.error(f"offload root was not cleaned after END_RUN: {root_dir}")
                    return False

        return True

    def _validate_checkpoint(self, run_dir: str, client_name: str) -> bool:
        checkpoint_name = DefaultCheckpointFileName.GLOBAL_MODEL
        checkpoint_paths = []
        for root, _, files in os.walk(run_dir):
            if checkpoint_name in files:
                checkpoint_paths.append(os.path.join(root, checkpoint_name))

        if not checkpoint_paths:
            self.logger.error(f"no {checkpoint_name} found for {client_name} under {run_dir}")
            return False

        for checkpoint_path in checkpoint_paths:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as e:
                self.logger.error(f"failed loading checkpoint {checkpoint_path}: {e}")
                return False

            weights = checkpoint.get("model", checkpoint)
            tensors = [value for value in weights.values() if isinstance(value, torch.Tensor)]
            if not tensors:
                self.logger.error(f"checkpoint has no tensor weights: {checkpoint_path}")
                return False
            if not all(
                torch.allclose(value, torch.full_like(value, self.expected_value), rtol=0.0, atol=1e-6)
                for value in tensors
            ):
                self.logger.error(
                    f"checkpoint {checkpoint_path} on {client_name} does not contain expected value "
                    f"{self.expected_value}"
                )
                return False

        return True
