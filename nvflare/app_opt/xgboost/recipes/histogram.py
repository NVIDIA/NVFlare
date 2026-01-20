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

from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


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
        per_site_config (dict, optional): Per-site configuration mapping site names to config dicts.
            Each config dict must contain 'data_loader' key with XGBDataLoader instance.
            Example: {"site-1": {"data_loader": CSVDataLoader(...)}, "site-2": {...}}

    Example:
        .. code-block:: python

            from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe
            from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
            from nvflare.recipe import SimEnv

            # Create recipe with per-site data loaders
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
                per_site_config={
                    "site-1": {"data_loader": CSVDataLoader(folder="/tmp/data/horizontal_xgb_data")},
                    "site-2": {"data_loader": CSVDataLoader(folder="/tmp/data/horizontal_xgb_data")},
                },
            )

            # Run simulation
            env = SimEnv()
            env.run(recipe)

    Note:
        - Data loaders must be configured via per_site_config parameter.
        - TensorBoard tracking is automatically configured for both server and clients.
        - Executor and metrics components are automatically added to all clients.
    """

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
        self.per_site_config = per_site_config

        # Configure the job
        self.job = self.configure()
        Recipe.__init__(self, self.job)

    def configure(self):
        """Configure the federated job for XGBoost histogram-based training."""
        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Configure controller and executor (histogram-based V2)
        from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
        from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor

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

        # Add executor to all clients
        executor_params = {
            "data_loader_id": self.data_loader_id,
            "metrics_writer_id": self.metrics_writer_id,
        }
        if self.secure:
            executor_params["in_process"] = True

        executor = FedXGBHistogramExecutor(**executor_params)
        job.to_clients(executor, id="xgb_executor")

        # Add TensorBoard receiver to server
        from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

        tb_receiver = TBAnalyticsReceiver(tb_folder="tb_events")
        job.to_server(tb_receiver, id="tb_receiver")

        # Add TensorBoard writer and event converter to all clients
        from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
        from nvflare.app_opt.tracking.tb.tb_writer import TBWriter

        metrics_writer = TBWriter(event_type="analytix_log_stats")
        job.to_clients(metrics_writer, id=self.metrics_writer_id)

        event_to_fed = ConvertToFedEvent(
            events_to_convert=["analytix_log_stats"],
            fed_event_prefix="fed.",
        )
        job.to_clients(event_to_fed, id="event_to_fed")

        # Add per-site data loaders if configured
        if self.per_site_config:
            for site_name, site_config in self.per_site_config.items():
                data_loader = site_config.get("data_loader")
                if data_loader is None:
                    raise ValueError(f"per_site_config for '{site_name}' must include 'data_loader' key")
                job.to(data_loader, site_name, id=self.data_loader_id)

        return job
