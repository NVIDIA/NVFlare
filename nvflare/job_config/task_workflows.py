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

"""Workflow compatibility checks for task-lifetime job export."""

from nvflare.app_common.workflows.cross_site_eval import CrossSiteEval
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.scaffold import Scaffold
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_common.workflows.scatter_and_gather_scaffold import ScatterAndGatherScaffold
from nvflare.job_config.fed_app_config import ServerAppConfig


def validate_task_workflows(server_app: ServerAppConfig):
    """Reject the workflow configurations not yet integrated with task workers."""
    for workflow in server_app.workflows:
        controller = workflow.controller
        aggregators = [getattr(controller, "aggregator", None)]
        if isinstance(controller, ScatterAndGather):
            # ScatterAndGather resolves its aggregator from the server components at startup.
            aggregators.append(server_app.components.get(controller.aggregator_id))

        reason = None
        if isinstance(controller, (Scaffold, ScatterAndGatherScaffold)):
            reason = "client SCAFFOLD control variates are not restored between workers"
        elif any(
            base.__module__ == "nvflare.app_opt.pt.fedce" and base.__name__ == "FedCEModelAggregator"
            for aggregator in aggregators
            for base in type(aggregator).__mro__
        ):
            # Inspect class ancestry without importing the optional PyTorch dependency.
            reason = "client FedCE previous-model and score-history state is not restored between workers"
        elif isinstance(controller, CrossSiteEval) or (
            isinstance(controller, CrossSiteModelEval) and controller._submit_model_task_name
        ):
            reason = "local-model submission is not implemented for fresh workers"
        if reason:
            raise ValueError(f"{type(controller).__name__} does not support task lifetime: {reason}; use job lifetime")
