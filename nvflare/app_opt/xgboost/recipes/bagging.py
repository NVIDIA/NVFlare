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

from typing import Optional

from pydantic import BaseModel, field_validator

from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.xgboost.tree_based.bagging_aggregator import XGBBaggingAggregator
from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor
from nvflare.app_opt.xgboost.tree_based.shareable_generator import XGBModelShareableGenerator
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _XGBBaggingValidator(BaseModel):
    # Allow custom types in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str
    min_clients: int
    training_mode: str
    num_rounds: int
    num_client_bagging: int
    num_local_parallel_tree: int
    local_subsample: float
    learning_rate: float
    objective: str
    max_depth: int
    eval_metric: str
    tree_method: str
    use_gpus: bool
    nthread: int
    lr_mode: str
    save_name: str
    data_loader_id: str

    @field_validator("training_mode")
    @classmethod
    def check_training_mode(cls, v):
        if v not in ["bagging", "cyclic"]:
            raise ValueError("training_mode must be 'bagging' or 'cyclic'")
        return v

    @field_validator("local_subsample")
    @classmethod
    def check_subsample(cls, v):
        if not 0 < v <= 1.0:
            raise ValueError("local_subsample must be between 0 and 1")
        return v

    @field_validator("lr_mode")
    @classmethod
    def check_lr_mode(cls, v):
        if v not in ["uniform", "scaled"]:
            raise ValueError("lr_mode must be 'uniform' or 'scaled'")
        return v


class XGBBaggingRecipe(Recipe):
    """XGBoost Tree-Based Recipe for federated learning (supports Bagging and Cyclic modes).

    This recipe implements tree-based federated XGBoost with two training modes:
    - **Bagging**: Each client trains a local sub-forest, aggregated on server (federated Random Forest)
    - **Cyclic**: Clients train sequentially in rounds, each contributing to the global model

    Args:
        name (str): Name of the federated job.
        min_clients (int): The minimum number of clients for the job.
        training_mode (str, optional): Training mode ("bagging" or "cyclic"). Default is "bagging".
        num_rounds (int, optional): Number of training rounds. Default is 1 for bagging, 100 for cyclic.
        num_client_bagging (int, optional): Number of clients for bagging. Default is min_clients.
        num_local_parallel_tree (int, optional): Number of parallel trees per client. Default is 5.
        local_subsample (float, optional): Subsample ratio for local training. Default is 0.8.
        learning_rate (float, optional): Learning rate for XGBoost. Default is 0.1.
        objective (str, optional): Learning objective. Default is "binary:logistic".
        max_depth (int, optional): Maximum tree depth. Default is 8.
        eval_metric (str, optional): Evaluation metric. Default is "auc".
        tree_method (str, optional): Tree construction method. Default is "hist".
        use_gpus (bool, optional): Whether to use GPUs. Default is False.
        nthread (int, optional): Number of threads. Default is 16.
        lr_mode (str, optional): Learning rate mode ("uniform" or "scaled"). Default is "uniform".
        save_name (str, optional): Model save name. Default is "xgboost_model.json".
        data_loader_id (str, optional): ID of the data loader component. Default is "dataloader".
        per_site_config (dict, optional): Per-site configuration mapping site names to config dicts.
            Each config dict must contain 'data_loader' key with XGBDataLoader instance.
            Can optionally include 'lr_scale' for scaled learning rate mode.
            Example: {"site-1": {"data_loader": CSVDataLoader(...), "lr_scale": 0.5}, "site-2": {...}}

    Example:
        .. code-block:: python

            from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
            from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
            from nvflare.recipe import SimEnv

            # Bagging mode (federated Random Forest) with uniform learning rate
            recipe = XGBBaggingRecipe(
                name="random_forest",
                min_clients=3,
                training_mode="bagging",
                num_rounds=1,
                num_local_parallel_tree=5,
                local_subsample=0.5,
                per_site_config={
                    "site-1": {"data_loader": CSVDataLoader(folder="/tmp/data")},
                    "site-2": {"data_loader": CSVDataLoader(folder="/tmp/data")},
                    "site-3": {"data_loader": CSVDataLoader(folder="/tmp/data")},
                },
            )

            # Or with scaled learning rate (data-size dependent)
            recipe = XGBBaggingRecipe(
                name="random_forest_scaled",
                min_clients=3,
                training_mode="bagging",
                lr_mode="scaled",
                per_site_config={
                    "site-1": {"data_loader": CSVDataLoader(folder="/tmp/data"), "lr_scale": 0.5},
                    "site-2": {"data_loader": CSVDataLoader(folder="/tmp/data"), "lr_scale": 0.3},
                    "site-3": {"data_loader": CSVDataLoader(folder="/tmp/data"), "lr_scale": 0.2},
                },
            )

            env = SimEnv(num_clients=3)
            run = recipe.execute(env)
    """

    def __init__(
        self,
        name: str,
        min_clients: int,
        training_mode: str = "bagging",
        num_rounds: Optional[int] = None,
        num_client_bagging: Optional[int] = None,
        num_local_parallel_tree: int = 5,
        local_subsample: float = 0.8,
        learning_rate: float = 0.1,
        objective: str = "binary:logistic",
        max_depth: int = 8,
        eval_metric: str = "auc",
        tree_method: str = "hist",
        use_gpus: bool = False,
        nthread: int = 16,
        lr_mode: str = "uniform",
        save_name: str = "xgboost_model.json",
        data_loader_id: str = "dataloader",
        data_loader: Optional["XGBDataLoader"] = None,
        per_site_config: Optional[dict[str, dict]] = None,
    ):
        # Set default num_rounds based on training_mode if not specified
        if num_rounds is None:
            num_rounds = 1 if training_mode == "bagging" else 100

        # Validate inputs internally
        v = _XGBBaggingValidator(
            name=name,
            min_clients=min_clients,
            training_mode=training_mode,
            num_rounds=num_rounds,
            num_client_bagging=num_client_bagging if num_client_bagging is not None else min_clients,
            num_local_parallel_tree=num_local_parallel_tree,
            local_subsample=local_subsample,
            learning_rate=learning_rate,
            objective=objective,
            max_depth=max_depth,
            eval_metric=eval_metric,
            tree_method=tree_method,
            use_gpus=use_gpus,
            nthread=nthread,
            lr_mode=lr_mode,
            save_name=save_name,
            data_loader_id=data_loader_id,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.training_mode = v.training_mode
        self.num_rounds = v.num_rounds
        self.num_client_bagging = v.num_client_bagging
        self.num_local_parallel_tree = v.num_local_parallel_tree
        self.local_subsample = v.local_subsample
        self.learning_rate = v.learning_rate
        self.objective = v.objective
        self.max_depth = v.max_depth
        self.eval_metric = v.eval_metric
        self.tree_method = v.tree_method
        self.use_gpus = v.use_gpus
        self.nthread = v.nthread
        self.lr_mode = v.lr_mode
        self.save_name = v.save_name
        self.data_loader_id = v.data_loader_id
        self.data_loader = data_loader
        self.per_site_config = per_site_config

        # Validate data loader configuration
        if data_loader is not None and per_site_config is not None:
            raise ValueError(
                "Cannot specify both 'data_loader' and 'per_site_config'. "
                "Use 'data_loader' for common config across all clients, "
                "or 'per_site_config' for site-specific configs."
            )

        if data_loader is None and per_site_config is None:
            raise ValueError(
                "Must provide either 'data_loader' or 'per_site_config'. "
                "Use 'data_loader=CSVDataLoader(...)' for common config, "
                'or \'per_site_config={"site-1": {"data_loader": ...}}\' for site-specific configs.'
            )

        # Configure the job
        self.job = self.configure()
        Recipe.__init__(self, self.job)

    def configure(self):
        """Configure the federated job for XGBoost tree-based training."""
        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Configure server components
        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            start_round=0,
            aggregator_id="aggregator",
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
            wait_time_after_min_received=0,
            train_timeout=0,
            allow_empty_global_weights=True,
            task_check_period=0.01,
            persist_every_n_rounds=0,
            snapshot_every_n_rounds=0,
        )
        job.to_server(controller, id="xgb_controller")

        persistor = XGBModelPersistor(save_name=self.save_name)
        job.to_server(persistor, id="persistor")

        shareable_generator = XGBModelShareableGenerator()
        job.to_server(shareable_generator, id="shareable_generator")

        aggregator = XGBBaggingAggregator()
        job.to_server(aggregator, id="aggregator")

        # Add executors and data loaders
        from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor

        if self.data_loader:
            # Common data loader for all clients
            # Create default executor for all clients
            executor = FedXGBTreeExecutor(
                data_loader_id=self.data_loader_id,
                training_mode=self.training_mode,
                num_client_bagging=self.num_client_bagging,
                num_local_parallel_tree=self.num_local_parallel_tree,
                local_subsample=self.local_subsample,
                local_model_path="model.json",
                global_model_path="model_global.json",
                learning_rate=self.learning_rate,
                objective=self.objective,
                max_depth=self.max_depth,
                eval_metric=self.eval_metric,
                tree_method=self.tree_method,
                use_gpus=self.use_gpus,
                nthread=self.nthread,
                lr_scale=1.0,
                lr_mode=self.lr_mode,
            )
            job.to_clients(executor, id="xgb_tree_executor")
            job.to_clients(self.data_loader, id=self.data_loader_id)
        elif self.per_site_config:
            # Site-specific executors and data loaders
            for site_name, site_config in self.per_site_config.items():
                data_loader = site_config.get("data_loader")
                if data_loader is None:
                    raise ValueError(f"per_site_config for '{site_name}' must include 'data_loader' key")

                # Get lr_scale from config, default to 1.0
                lr_scale = site_config.get("lr_scale", 1.0)

                # Create executor for this site
                executor = FedXGBTreeExecutor(
                    data_loader_id=self.data_loader_id,
                    training_mode=self.training_mode,
                    num_client_bagging=self.num_client_bagging,
                    num_local_parallel_tree=self.num_local_parallel_tree,
                    local_subsample=self.local_subsample,
                    local_model_path="model.json",
                    global_model_path="model_global.json",
                    learning_rate=self.learning_rate,
                    objective=self.objective,
                    max_depth=self.max_depth,
                    eval_metric=self.eval_metric,
                    tree_method=self.tree_method,
                    use_gpus=self.use_gpus,
                    nthread=self.nthread,
                    lr_scale=lr_scale,
                    lr_mode=self.lr_mode,
                )
                job.to(executor, site_name, id="xgb_tree_executor")
                job.to(data_loader, site_name, id=self.data_loader_id)

        return job
