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

from typing import Any, Dict, List, Optional, Union

from nvflare.app_opt.pt.fedsm import FedSM, FedSMModelAggregator, PTFedSMModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class FedSMRecipe(Recipe):
    """PyTorch recipe for personalized federated learning with FedSM.

    FedSM jointly trains a global model, one personalized model per client, and
    a selector model. Client scripts receive a model bundle and must return a
    FULL bundle containing global and selector weight differences plus the full
    personalized model. They may also return selector optimizer state.

    Unlike FedAvg, FedSM requires the complete site set before job construction.
    The recipe creates one client app per site, and every configured site
    participates in every round.
    """

    def __init__(
        self,
        *,
        name: str = "fedsm",
        model: Union[Any, Dict[str, Any]],
        selector_model: Union[Any, Dict[str, Any]],
        sites: List[str],
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        site_label_mapping: Optional[Dict[str, int]] = None,
        soft_pull_lambda: float = 0.7,
        initial_ckpt: Optional[str] = None,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.PYTORCH,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        server_memory_gc_rounds: int = 0,
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ):
        if model is None:
            raise ValueError("FedSMRecipe requires model")
        if selector_model is None:
            raise ValueError("FedSMRecipe requires selector_model")
        if not isinstance(sites, list) or not sites:
            raise ValueError("sites must be a non-empty list of unique site names")
        if any(not isinstance(site, str) or not site for site in sites):
            raise ValueError("sites must contain non-empty strings")
        if len(sites) != len(set(sites)):
            raise ValueError("sites must be a non-empty list of unique site names")
        if min_clients != len(sites):
            raise ValueError(
                "FedSMRecipe currently requires min_clients to equal len(sites) "
                "so every personalized model participates in each SoftPull update"
            )
        if server_expected_format != ExchangeFormat.PYTORCH:
            raise ValueError("FedSMRecipe requires server_expected_format=ExchangeFormat.PYTORCH")

        mapping = site_label_mapping or {site: index for index, site in enumerate(sites)}
        if set(mapping) != set(sites):
            raise ValueError("site_label_mapping keys must exactly match sites")
        if set(mapping.values()) != set(range(len(sites))):
            raise ValueError("site_label_mapping values must be contiguous selector labels starting at 0")

        from nvflare.recipe.utils import prepare_initial_ckpt, recipe_model_to_job_model, validate_ckpt

        validate_ckpt(initial_ckpt)
        if isinstance(model, dict):
            model = recipe_model_to_job_model(model)
        if isinstance(selector_model, dict):
            selector_model = recipe_model_to_job_model(selector_model)

        self.name = name
        self.model = model
        self.selector_model = selector_model
        self.sites = list(sites)
        self.min_clients = min_clients
        self.num_rounds = num_rounds
        self.train_script = train_script
        self.train_args = train_args
        self.site_label_mapping = dict(mapping)
        self.soft_pull_lambda = soft_pull_lambda
        self.initial_ckpt = initial_ckpt
        self.server_expected_format = server_expected_format

        job = BaseFedJob(name=name, min_clients=min_clients)
        ckpt_path = prepare_initial_ckpt(initial_ckpt, job)
        persistor = PTFedSMModelPersistor(
            model=model,
            selector_model=selector_model,
            client_ids=sites,
            source_ckpt_file_full_name=ckpt_path,
        )
        persistor_id = job.to_server(persistor, id="persistor")

        aggregator = FedSMModelAggregator(soft_pull_lambda=soft_pull_lambda)
        controller = FedSM(
            num_clients=min_clients,
            num_rounds=num_rounds,
            persistor_id=persistor_id,
            client_id_label_mapping=mapping,
            aggregator=aggregator,
            memory_gc_rounds=server_memory_gc_rounds,
        )
        job.to_server(controller)

        for site in sites:
            executor = ScriptRunner(
                script=train_script,
                script_args=train_args,
                launch_external_process=launch_external_process,
                command=command,
                framework=FrameworkType.PYTORCH,
                server_expected_format=server_expected_format,
                params_transfer_type=TransferType.FULL,
                launch_once=launch_once,
                shutdown_timeout=shutdown_timeout,
                memory_gc_rounds=client_memory_gc_rounds,
                cuda_empty_cache=cuda_empty_cache,
            )
            job.to(executor, site)

        super().__init__(job)

    def configured_sites(self) -> List[str]:
        if self._helper_per_site_config is not None:
            return super().configured_sites()
        return list(self.sites)
