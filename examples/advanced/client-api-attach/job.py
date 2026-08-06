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

"""Submit and monitor a one-client NumPy FedAvg job that waits for an attached trainer."""

import argparse
from pathlib import Path

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import ProdEnv


class AttachNumpyFedAvgRecipe(NumpyFedAvgRecipe):
    def __init__(self, attach_id: str, allow_insecure_attach: bool, **kwargs):
        self.attach_id = attach_id
        self.allow_insecure_attach = allow_insecure_attach
        super().__init__(**kwargs)

    def _create_client_runner(self, site_config):
        return ClientAPIExecutor(
            execution_mode="attach",
            attach_id=self.attach_id,
            attach_timeout=300.0,
            heartbeat_interval=5.0,
            heartbeat_timeout=30.0,
            task_wait_timeout=600.0,
            allow_insecure_attach=self.allow_insecure_attach,
            params_exchange_format=ExchangeFormat.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.FULL,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach_id", default="numpy_trainer")
    parser.add_argument("--startup_kit_location", required=True)
    parser.add_argument("--username", default="admin@nvidia.com")
    parser.add_argument(
        "--allow_insecure_attach",
        action="store_true",
        help="Permit an unprotected CJ-owned network listener for a trusted development/POC environment",
    )
    args = parser.parse_args()

    recipe = AttachNumpyFedAvgRecipe(
        attach_id=args.attach_id,
        allow_insecure_attach=args.allow_insecure_attach,
        name="client-api-attach",
        min_clients=1,
        num_rounds=3,
        model=[[1, 2, 3], [4, 5, 6]],
        key_metric="weight_mean",
        # The recipe requires an entry-point resource, but Attach never launches it.
        # The operator starts trainer.py independently with its connection profile.
        train_script=str(Path(__file__).with_name("trainer.py")),
    )
    env = ProdEnv(
        startup_kit_location=args.startup_kit_location,
        username=args.username,
    )
    run = recipe.execute(env)
    print(f"Submitted {recipe.name!r} as job {run.get_job_id()}")
    result = run.get_result()
    print(f"Result: {result}")
    print(f"Status: {run.get_status()}")


if __name__ == "__main__":
    main()
