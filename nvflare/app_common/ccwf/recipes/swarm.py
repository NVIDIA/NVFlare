# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, CrossSiteEvalConfig, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class BaseSwarmLearningRecipe(Recipe):

    def __init__(
        self,
        name: str,
        server_config: SwarmServerConfig,
        client_config: SwarmClientConfig,
        cse_config: CrossSiteEvalConfig = None,
    ):
        job = CCWFJob(name=name)
        job.add_swarm(
            server_config=server_config,
            client_config=client_config,
            cse_config=cse_config,
        )
        Recipe.__init__(self, job)


class SimpleSwarmLearningRecipe(BaseSwarmLearningRecipe):

    def __init__(
        self,
        name: str,
        initial_model,
        num_rounds: int,
        train_script: str,
        train_args: dict = None,
        do_cross_site_eval: bool = False,
        cross_site_eval_timeout: float = 300,
    ):
        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
        if do_cross_site_eval:
            cse_config = CrossSiteEvalConfig(eval_task_timeout=cross_site_eval_timeout)
        else:
            cse_config = None

        if not train_args:
            train_args = {}

        server_config = SwarmServerConfig(num_rounds=num_rounds)
        client_config = SwarmClientConfig(
            executor=ScriptRunner(script=train_script, **train_args),
            aggregator=aggregator,
            persistor=PTFileModelPersistor(model=initial_model),
            shareable_generator=SimpleModelShareableGenerator(),
        )

        BaseSwarmLearningRecipe.__init__(self, name, server_config, client_config, cse_config)
