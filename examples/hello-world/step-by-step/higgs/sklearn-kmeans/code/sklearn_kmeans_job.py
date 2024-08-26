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
from kmeans_assembler import KMeansAssembler

from nvflare import FedJob
from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 3
    num_rounds = 20
    train_script = "code/kmeans_fl.py"
    script_args = "--data_root_dir /tmp/nvflare/dataset/output"

    aggregator_id = "aggregator"
    persistor_id = "persistor"
    shareable_generator_id = "shareable_generator"
    assembler_id = "kmeans_assembler"

    job = FedJob("sklearn_kmeans")

    initial_params = dict(n_clusters=2)
    job.to(JoblibModelParamPersistor(initial_params=initial_params), "server", id=persistor_id)
    job.to(FullModelShareableGenerator(), "server", id=shareable_generator_id)
    job.to(CollectAndAssembleAggregator(assembler_id=assembler_id), "server", id=aggregator_id)
    job.to(KMeansAssembler(), "server", id=assembler_id)

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
    job.simulator_run("/tmp/nvflare/sklearn_kmeans", gpu="0")
