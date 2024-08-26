# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 3
    num_rounds = 30
    train_script = "code/sgd_fl.py"
    script_args = "--data_root_dir /tmp/nvflare/dataset/output"

    aggregator_id = "aggregator"
    persistor_id = "persistor"
    shareable_generator_id = "shareable_generator"

    job = FedJob("sklearn_sgd")

    initial_params = dict(
        n_classes=2, learning_rate="constant", eta0=1e-05, loss="log_loss", penalty="l2", fit_intercept=True, max_iter=1
    )
    job.to(JoblibModelParamPersistor(initial_params=initial_params), "server", id=persistor_id)
    job.to(FullModelShareableGenerator(), "server", id=shareable_generator_id)
    job.to(InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), "server", id=aggregator_id)

    ctrl = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        start_round=0,
        wait_time_after_min_received=0,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",
        train_timeout=0,
        allow_empty_global_weights=True,
    )

    job.to(ctrl, "server")

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(script=train_script, script_args=script_args, framework=FrameworkType.RAW)
        job.to(runner, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs")
    job.simulator_run("/tmp/nvflare/sklearn_sgd", gpu="0")
