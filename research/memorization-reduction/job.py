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

"""Run two CPU clients for two rounds; reports demonstrate wiring, not efficacy."""

from model import SyntheticAuditor, create_model

from nvflare.app_common.filters.memorization_audit_filter import AuditSnapshotFilter, MemorizationAuditFilter
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def create_recipe():
    recipe = FedAvgRecipe(
        name="memorization-reduction",
        min_clients=2,
        num_rounds=2,
        model=create_model(),
        train_script="client.py",
        key_metric="ce",
        key_metric_mode="min",
    )
    recipe.add_client_input_filter(AuditSnapshotFilter(), tasks=["train"])
    recipe.add_client_output_filter(
        MemorizationAuditFilter(auditor=SyntheticAuditor(max_seconds=30), frequency=1), tasks=["train"]
    )
    return recipe


if __name__ == "__main__":
    run = create_recipe().execute(SimEnv(num_clients=2))
    print("Functional simulation artifacts:", run.get_result())
