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


def create_classes_table_static():
    from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
    from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
    from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
    from nvflare.app_common.ccwf import (
        CrossSiteEvalClientController,
        CrossSiteEvalServerController,
        CyclicClientController,
        SwarmClientController,
        SwarmServerController,
    )
    from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer
    from nvflare.app_common.response_processors.global_weights_initializer import GlobalWeightsInitializer
    from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
    from nvflare.app_common.workflows.cross_site_eval import CrossSiteEval
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.app_common.workflows.cyclic_ctl import CyclicController
    from nvflare.app_common.workflows.global_model_eval import GlobalModelEval
    from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
    from nvflare.app_common.workflows.scatter_and_gather_scaffold import ScatterAndGatherScaffold
    from nvflare.app_opt.pt import PTFileModelLocator, PTFileModelPersistor

    classes = {
        ScatterAndGather,
        ScatterAndGatherScaffold,
        CollectAndAssembleAggregator,
        CrossSiteEval,
        CrossSiteEvalClientController,
        CrossSiteEvalServerController,
        CrossSiteModelEval,
        CyclicClientController,
        CyclicController,
        DXOAggregator,
        GlobalModelEval,
        GlobalWeightsInitializer,
        Gatherer,
        SwarmClientController,
        SwarmServerController,
        FullModelShareableGenerator,
        InTimeAccumulateWeightedAggregator,
        PTFileModelPersistor,
        PTFileModelLocator,
    }

    class_table = {}
    for item in classes:
        class_table[item.__name__] = item.__module__
    return class_table
