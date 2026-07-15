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

from typing import Optional

from pydantic import BaseModel, field_validator

from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.app_opt.tracking.tb.tb_writer import TBWriter
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import (
    _apply_legacy_constructor_config,
    _configure_per_site_clients,
    _validate_per_site_targets,
)


# Internal — not part of the public API
class _XGBHistogramValidator(BaseModel):

    # Allow custom types in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str
    min_clients: int
    num_rounds: int
    early_stopping_rounds: int
    use_gpus: bool
    secure: bool
    client_ranks: dict
    xgb_params: dict
    data_loader_id: str
    metrics_writer_id: str

    @field_validator("num_rounds")
    @classmethod
    def check_num_rounds(cls, v):
        if v < 1:
            raise ValueError("num_rounds must be at least 1")
        return v


class XGBHorizontalRecipe(Recipe):
    """XGBoost Horizontal Federated Learning Recipe.

    Recipe parameters, including ``xgb_params`` and nested ``per_site_config`` values,
    must never contain actual secrets. Read secrets from site environment variables or mounted
    files; references are supported only where documented in :mod:`nvflare.recipe.secrets`.

    This recipe implements horizontal federated XGBoost using histogram-based algorithms.
    In horizontal federated learning, each client has different samples with the same features.
    The histogram-based approach enables efficient gradient boosting by computing histograms
    of gradients and hessians collaboratively across clients.

    Args:
        name (str): Name of the federated job.
        min_clients (int): The minimum number of clients for the job.
        num_rounds (int): Number of boosting rounds.
        early_stopping_rounds (int, optional): Early stopping rounds. Default is 2.
        use_gpus (bool, optional): Whether to use GPUs for training. Default is False.
        secure (bool, optional): Enable secure training with Homomorphic Encryption (HE). Default is False.
            Requires encryption plugins to be installed and configured.
            When secure=True, client_ranks must be provided.
        client_ranks (dict, optional): Mapping of client names to ranks for secure training.
            Required when secure=True. Maps each client name to a unique rank (0-indexed).
            Example: {"site-1": 0, "site-2": 1, "site-3": 2}.
        xgb_params (dict, optional): XGBoost parameters passed to xgboost.train(). If None, uses default params.
            Default params: max_depth=8, eta=0.1, objective='binary:logistic', eval_metric='auc',
            tree_method='hist', nthread=16.
        data_loader_id (str, optional): ID of the data loader component. Default is 'dataloader'.
        metrics_writer_id (str, optional): ID of the metrics writer component. Default is 'metrics_writer'.
        per_site_config (dict, optional): Deprecated constructor form of per-site configuration.
            New code should call ``set_per_site_config(recipe, config)`` immediately after construction.

    Example:
        .. code-block:: python

            from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe
            from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
            from nvflare.recipe import SimEnv, set_per_site_config

            # Build per-site configuration with data loaders
            per_site_config = {
                "site-1": {"data_loader": CSVDataLoader(folder="/tmp/data/horizontal_xgb_data")},
                "site-2": {"data_loader": CSVDataLoader(folder="/tmp/data/horizontal_xgb_data")},
            }

            # Create recipe
            recipe = XGBHorizontalRecipe(
                name="xgb_higgs_horizontal",
                min_clients=2,
                num_rounds=100,
                xgb_params={
                    "max_depth": 8,
                    "eta": 0.1,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                },
            )
            set_per_site_config(recipe, per_site_config)

            # Run simulation with explicit client list
            clients = list(per_site_config.keys())
            env = SimEnv(clients=clients)
            run = recipe.execute(env)

    Note:
        - Data loaders must be configured with ``set_per_site_config`` before export or execution.
        - TensorBoard tracking is automatically configured for the server and configured sites.
        - Executor and metrics components are automatically added to each configured site.
    """

    _UNSUPPORTED_SECRET_REF_ATTRS = Recipe._UNSUPPORTED_SECRET_REF_ATTRS | {"per_site_config"}

    def __init__(
        self,
        name: str,
        min_clients: int,
        num_rounds: int,
        early_stopping_rounds: int = 2,
        use_gpus: bool = False,
        secure: bool = False,
        client_ranks: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        data_loader_id: str = "dataloader",
        metrics_writer_id: str = "metrics_writer",
        per_site_config: Optional[dict[str, dict]] = None,
    ):
        # Set default XGBoost params if not provided
        if xgb_params is None:
            xgb_params = {
                "max_depth": 8,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "nthread": 16,
            }

        # Validate inputs internally
        v = _XGBHistogramValidator(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            early_stopping_rounds=early_stopping_rounds,
            use_gpus=use_gpus,
            secure=secure,
            client_ranks=client_ranks if client_ranks else {},
            xgb_params=xgb_params,
            data_loader_id=data_loader_id,
            metrics_writer_id=metrics_writer_id,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.early_stopping_rounds = v.early_stopping_rounds
        self.use_gpus = v.use_gpus
        self.secure = v.secure
        self.client_ranks = v.client_ranks
        self.xgb_params = v.xgb_params
        self.data_loader_id = v.data_loader_id
        self.metrics_writer_id = v.metrics_writer_id
        legacy_per_site_config = per_site_config
        self.per_site_config = None

        # Configure site-independent job components first. Site apps are added
        # only through the canonical per-site configuration hook.
        self.job = self.configure()
        Recipe.__init__(self, self.job)

        if legacy_per_site_config is not None:
            _apply_legacy_constructor_config(self, legacy_per_site_config)

    def configure(self):
        """Configure the federated job for XGBoost histogram-based training."""
        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)
        job.to_server(MetricsArtifactWriter(), id="metrics_artifact_writer")

        # Configure controller and executor (histogram-based V2)
        controller_kwargs = {
            "num_rounds": self.num_rounds,
            "data_split_mode": 0,  # 0 = horizontal
            "secure_training": self.secure,
            "xgb_options": {"early_stopping_rounds": self.early_stopping_rounds, "use_gpus": self.use_gpus},
            "xgb_params": self.xgb_params,
        }

        # Add client_ranks if secure training is enabled
        if self.secure and self.client_ranks:
            controller_kwargs["client_ranks"] = self.client_ranks
            controller_kwargs["in_process"] = True  # Required for secure training

        controller = XGBFedController(**controller_kwargs)
        job.to_server(controller, id="xgb_controller")

        # Add TensorBoard receiver to server
        tb_receiver = TBAnalyticsReceiver(tb_folder="tb_events")
        job.to_server(tb_receiver, id="tb_receiver")

        return job

    def _add_client_components(self, job: FedJob, site_name: str, site_config: dict) -> None:
        executor_params = {
            "data_loader_id": self.data_loader_id,
            "metrics_writer_id": self.metrics_writer_id,
        }
        if self.secure:
            executor_params["in_process"] = True

        job.to(FedXGBHistogramExecutor(**executor_params), site_name, id="xgb_executor")
        job.to(TBWriter(event_type="analytix_log_stats"), site_name, id=self.metrics_writer_id)
        job.to(
            ConvertToFedEvent(events_to_convert=["analytix_log_stats"], fed_event_prefix="fed."),
            site_name,
            id="event_to_fed",
        )
        job.to(site_config["data_loader"], site_name, id=self.data_loader_id)

    def _apply_per_site_config(self, config: dict[str, dict]) -> None:
        _validate_per_site_targets(config, self.min_clients)
        for site_name, site_config in config.items():
            if site_config.get("data_loader") is None:
                raise ValueError(f"per_site_config for {site_name!r} must include 'data_loader' key")

        _configure_per_site_clients(
            self.job,
            config,
            self._add_client_components,
            replace_all_clients=False,
        )
        self.per_site_config = dict(config)

    def _validate_before_use(self) -> None:
        if not self.configured_sites():
            raise RuntimeError("XGBHorizontalRecipe requires set_per_site_config() before export or execution")
