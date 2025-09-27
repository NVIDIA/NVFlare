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

import json
import os

import torch
import torch.optim as optim

# Import NVFLARE-specific data utilities
from data_utils.data_loader import get_client_data_partition, get_dataset
from fedhca2_core.datasets.custom_transforms import get_transformations
from fedhca2_core.datasets.utils.configs import TEST_SCALE, TRAIN_SCALE
from fedhca2_core.losses import get_criterion

# Import original FedHCA2 components
from fedhca2_core.models.build_models import build_model
from fedhca2_core.train_utils import local_train as fedhca2_local_train
from fedhca2_core.utils import RunningMeter
from timm.scheduler.cosine_lr import CosineLRScheduler

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants


class FedHCA2Learner(Executor):
    """NVFLARE executor for FedHCA2 federated multi-task learning"""

    def __init__(
        self,
        train_config_filename="config_train.json",
        aggregation_epochs=1,
        analytic_sender_id="analytic_sender",
    ):
        super().__init__()

        self.train_config_filename = train_config_filename
        self.aggregation_epochs = aggregation_epochs
        self.analytic_sender_id = analytic_sender_id

        # Configuration
        self.config_info = None
        self.client_config = None

        # Training hyperparameters
        self.lr = None
        self.weight_decay = None
        self.local_epochs = None
        self.batch_size = None
        self.val_batch_size = None
        self.warmup_epochs = None
        self.optimizer_name = None
        self.nworkers = None
        self.fp16 = None

        # Model configuration
        self.backbone_type = None
        self.backbone_pretrained = None

        # Training components
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.train_loss = None
        self.writer = None

        # Client information
        self.client_name = None
        self.tasks = None
        self.dataname = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the learner with client-specific configuration"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_name = fl_ctx.get_identity_name()

        print(f"Initializing FedHCA2Learner for {self.client_name}")

        # Load training configuration using standard NVFLARE pattern
        self.train_config(fl_ctx)

        # Load client-specific configuration
        self.client_config = self._get_client_config_for_site(self.client_name)

        # Extract client info
        self.tasks = self.client_config.get('tasks')
        self.dataname = self.client_config.get('dataname')

        # Load client-specific training hyperparameters
        self.lr = self.client_config.get('learning_rate', 0.0001)
        self.weight_decay = self.client_config.get('weight_decay', 0.0001)
        self.local_epochs = self.client_config.get('local_epochs', 1)
        self.batch_size = self.client_config.get('batch_size', 4)
        self.val_batch_size = self.batch_size
        self.warmup_epochs = self.client_config.get('warmup_epochs', 5)
        self.optimizer_name = self.client_config.get('optimizer', 'adamw')
        self.nworkers = self.client_config.get('nworkers', 4)
        self.fp16 = self.client_config.get('fp16', True)

        if self.tasks is None:
            raise ValueError(f"Tasks not found in config for client: {self.client_name}")
        if self.dataname is None:
            raise ValueError(f"Dataname not found in config for client: {self.client_name}")

        print(f"Initializing {self.client_name}: {self.dataname} dataset, tasks={self.tasks}")

        # Build model
        self.model = build_model(
            tasks=self.tasks,
            dataname=self.dataname,
            backbone_type=self.backbone_type,
            backbone_pretrained=self.backbone_pretrained,
        ).to(self.device)

        # Build optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Build scheduler
        max_epochs = 100 * self.local_epochs
        self.scheduler = CosineLRScheduler(
            optimizer=self.optimizer,
            t_initial=max_epochs - self.warmup_epochs,
            lr_min=1.25e-6,
            warmup_t=self.warmup_epochs,
            warmup_lr_init=1.25e-7,
            warmup_prefix=True,
        )

        # Build criterion
        self.criterion = get_criterion(self.dataname, self.tasks).to(self.device)

        # Build scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # Build data loaders
        self._setup_data_loaders(fl_ctx)

        # Initialize TensorBoard writer
        self.writer = parts.get(self.analytic_sender_id)  # Use NVFLARE AnalyticsSender if available
        if not self.writer:  # Fallback to local TensorBoard writer
            from torch.utils.tensorboard import SummaryWriter

            engine = fl_ctx.get_engine()
            ws = engine.get_workspace()
            app_dir = ws.get_app_dir(fl_ctx.get_job_id())
            self.writer = SummaryWriter(app_dir)

        # Initialize loss meters
        self.train_loss = {task: RunningMeter() for task in self.tasks}

        print(f"{self.client_name} initialized successfully")

    def train_config(self, fl_ctx: FLContext):
        """Load training configuration using standard NVFLARE pattern"""
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)

        if not os.path.isfile(train_config_file_path):
            raise FileNotFoundError(f"Training configuration file does not exist at {train_config_file_path}")

        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        # Load model configuration
        model_config = self.config_info["model"]
        self.backbone_type = model_config["backbone_type"]
        self.backbone_pretrained = model_config["backbone_pretrained"]

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """Execute different tasks"""
        if task_name == "train":
            return self.train(shareable, fl_ctx)
        elif task_name == "validate":
            return self.validate(shareable, fl_ctx)
        elif task_name == "submit_model":
            return self.submit_model(shareable, fl_ctx)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def train(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Train the local model using original FedHCA2 logic"""
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        if current_round is None:
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, 0)

        # Ensure initialization
        if self.config_info is None or self.tasks is None or self.client_name is None:
            self.initialize({}, fl_ctx)

        print(f"Round {current_round}: Training {self.client_name} on {self.tasks}")

        # Update model with received parameters
        self._update_local_model(shareable)

        # Reset loss meters
        for task in self.tasks:
            self.train_loss[task].reset()

        # Original FedHCA2 local training
        fedhca2_local_train(
            idx=0,
            cr=current_round if current_round is not None else 0,
            local_epochs=self.local_epochs,
            tasks=self.tasks,
            train_dl=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            scaler=self.scaler,
            train_loss=self.train_loss,
            local_rank=0,
            fp16=self.fp16,
            writer=self.writer,
        )

        # Log training results
        for task, loss_meter in self.train_loss.items():
            print(f"{task} loss: {loss_meter.avg:.4f}")
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(
                    f"train_loss/{task}", loss_meter.avg, current_round if current_round is not None else 0
                )

        # Perform evaluation every 5 rounds
        if current_round is not None and current_round % 5 == 0:
            print(f"Evaluating {self.client_name} at round {current_round}")
            eval_results = self._perform_evaluation()
            if eval_results:
                print(f"Evaluation results:")
                for task, task_metrics in eval_results.items():
                    if isinstance(task_metrics, dict):
                        for metric_name, metric_value in task_metrics.items():
                            print(f"{task}_{metric_name}: {metric_value:.4f}")
                            # Log to TensorBoard
                            if self.writer:
                                self.writer.add_scalar(f"eval/{task}_{metric_name}", metric_value, current_round)
                    else:
                        print(f"{task}: {task_metrics:.4f}")
                        # Log to TensorBoard
                        if self.writer:
                            self.writer.add_scalar(f"eval/{task}", task_metrics, current_round)

        return self._create_model_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Validation task"""
        if self.config_info is None or self.tasks is None or self.client_name is None:
            self.initialize({}, fl_ctx)

        return self._run_validation(shareable, fl_ctx)

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Submit the current model"""
        if self.config_info is None or self.tasks is None or self.client_name is None:
            self.initialize({}, fl_ctx)

        return self._create_model_shareable()

    def _run_validation(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Run validation with FedHCA2 evaluation logic"""
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND) + 1
        if current_round is None:
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, 0) + 1

        # Update model with received parameters
        self._update_local_model(shareable)

        # Check if we should perform evaluation (every 5 rounds)
        if current_round is not None and current_round % 5 == 0:
            eval_results = self._perform_evaluation()
            # Log validation results to TensorBoard
            if eval_results and self.writer:
                for task, task_metrics in eval_results.items():
                    if isinstance(task_metrics, dict):
                        for metric_name, metric_value in task_metrics.items():
                            self.writer.add_scalar(f"val/{task}_{metric_name}", metric_value, current_round)
                    else:
                        self.writer.add_scalar(f"val/{task}", task_metrics, current_round)
            return self._create_validation_result(eval_results)
        else:
            return self._create_validation_result()

    def _perform_evaluation(self) -> dict:
        """Perform comprehensive evaluation using fedhca_core evaluation logic"""
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            return {}

        # Import evaluation utilities from fedhca_core
        from fedhca2_core.evaluation.evaluate_utils import PerformanceMeter

        # Setup evaluation meter for this client's tasks
        eval_meter = PerformanceMeter(self.dataname, self.tasks)
        try:
            eval_meter.reset()
        except AttributeError:
            eval_meter = PerformanceMeter(self.dataname, self.tasks)

        # Set model to evaluation mode
        self.model.eval()
        eval_results = {}

        with torch.no_grad():
            try:
                for batch_idx, batch in enumerate(self.val_loader):
                    if batch_idx >= 10:  # Limit evaluation to 10 batches for efficiency
                        break

                    # Move batch to device
                    from fedhca2_core.utils import get_output, to_cuda

                    batch = to_cuda(batch)
                    inputs = batch['image']

                    # Forward pass
                    outputs = self.model(inputs)

                    # Transform outputs using get_output
                    processed_outputs = {t: get_output(outputs[t], t) for t in self.tasks}

                    # Update evaluation meter
                    try:
                        eval_meter.update(processed_outputs, batch)
                    except (RuntimeError, ValueError):
                        continue

                # Get final evaluation scores
                try:
                    eval_results = eval_meter.get_score()
                except Exception:
                    eval_results = {task: 0.0 for task in self.tasks}

            except Exception:
                eval_results = {task: 0.0 for task in self.tasks}

        # Set model back to training mode
        self.model.train()
        return eval_results

    def _create_validation_result(self, eval_results=None) -> Shareable:
        """Create validation result shareable"""
        from nvflare.apis.dxo import DXO, DataKind, MetaKey

        if eval_results is None:
            eval_results = {}

        # Create validation DXO
        dxo = DXO(data_kind=DataKind.METRICS, data=eval_results)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)

        return dxo.to_shareable()

    def _get_client_config_for_site(self, client_name: str) -> dict:
        """Get client-specific configuration based on client mapping in config file"""

        # Get client mapping from config
        client_configs = self.config_info.get('client_configs', {})
        if client_name not in client_configs:
            raise ValueError(
                f"No configuration found for client '{client_name}'. "
                f"Available clients: {list(client_configs.keys())}"
            )

        client_mapping = client_configs[client_name]
        dataset_name = client_mapping['dataset']  # 'pascalcontext' or 'nyud'
        client_tasks = client_mapping['tasks']  # List of tasks for this client

        # Get dataset-specific training parameters
        dataset_configs = self.config_info.get('dataset_configs', {})
        if dataset_name not in dataset_configs:
            raise ValueError(f"Dataset configuration not found: {dataset_name}")

        dataset_config = dataset_configs[dataset_name]

        # Build client configuration
        client_config = {
            'dataname': dataset_name,
            'tasks': client_tasks,
            'is_single_task': len(client_tasks) == 1,  # Infer from number of tasks
            'client_id': self._get_client_index(client_name, client_configs),
        }

        # Add training hyperparameters from dataset config
        for key in [
            'learning_rate',
            'weight_decay',
            'local_epochs',
            'batch_size',
            'warmup_epochs',
            'optimizer',
            'nworkers',
            'fp16',
        ]:
            if key in dataset_config:
                client_config[key] = dataset_config[key]

        return client_config

    def _get_client_index(self, client_name: str, client_configs: dict) -> int:
        """Get client index based on client name ordering"""
        client_names = sorted(client_configs.keys())
        try:
            return client_names.index(client_name)
        except ValueError:
            raise ValueError(f"Client '{client_name}' not found in client configurations")

    def _get_experiment_config_from_training_config(self) -> dict:
        """Extract experiment config from training config"""
        # Build legacy format for data partitioning compatibility
        st_datasets_by_name = {}
        mt_datasets_by_name = {}

        # Group clients by dataset and task count
        for client_name, client_mapping in self.config_info.get('client_configs', {}).items():
            dataset_name = client_mapping['dataset']
            tasks = client_mapping['tasks']
            dataset_config = self.config_info['dataset_configs'][dataset_name]

            if len(tasks) == 1:  # Single-task
                if dataset_name not in st_datasets_by_name:
                    st_datasets_by_name[dataset_name] = {
                        'dataname': dataset_name,
                        'task_dict': {},
                        **dataset_config,
                    }
                # Add this client's task to the task_dict
                st_datasets_by_name[dataset_name]['task_dict'][tasks[0]] = 1

            else:  # Multi-task
                if dataset_name not in mt_datasets_by_name:
                    mt_datasets_by_name[dataset_name] = {
                        'dataname': dataset_name,
                        'task_dict': {task: 1 for task in tasks},
                        'client_num': 0,
                        **dataset_config,
                    }
                # Count number of MT clients for this dataset
                mt_datasets_by_name[dataset_name]['client_num'] += 1

        # Convert to lists
        st_datasets = list(st_datasets_by_name.values())
        mt_datasets = list(mt_datasets_by_name.values())

        return {
            "ST_Datasets": st_datasets,
            "MT_Datasets": mt_datasets,
        }

    def _setup_data_loaders(self, fl_ctx):
        """Setup data loaders with client-specific partitioning"""
        # Get transforms
        train_transform = get_transformations(TRAIN_SCALE[self.dataname], train=True)
        val_transform = get_transformations(TEST_SCALE[self.dataname], train=False)

        # Get experiment config for data partitioning
        exp_config = self._get_experiment_config_from_training_config()

        # Get client-specific data partition
        data_indices = get_client_data_partition(self.client_config['client_id'], self.client_config, exp_config)

        # Create datasets
        train_dataset = get_dataset(
            dataname=self.dataname, tasks=self.tasks, train=True, dataidxs=data_indices, transform=train_transform
        )

        val_dataset = get_dataset(dataname=self.dataname, tasks=self.tasks, train=False, transform=val_transform)

        # Create data loaders
        from fedhca2_core.datasets.utils.custom_collate import collate_mil
        from torch.utils import data

        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            collate_fn=collate_mil,
            drop_last=True,
        )

        self.val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.nworkers,
            collate_fn=collate_mil,
        )

    def _update_local_model(self, shareable: Shareable):
        """Update local model with received parameters"""
        if not shareable:
            return

        # Convert shareable to FLModel
        from nvflare.app_common.utils.fl_model_utils import FLModelUtils

        fl_model = FLModelUtils.from_shareable(shareable, None)

        if not fl_model or not fl_model.params:
            return

        # Check for personalized models (FedHCA2 case)
        if fl_model.meta and "personalized_models" in fl_model.meta:
            personalized_models = fl_model.meta["personalized_models"]
            if self.client_name in personalized_models:
                params = personalized_models[self.client_name]
                state_dict = {k: torch.tensor(v) for k, v in params.items()}
                self.model.load_state_dict(state_dict, strict=False)
                return

        # Load global model
        state_dict = {k: torch.tensor(v) for k, v in fl_model.params.items()}
        self.model.load_state_dict(state_dict, strict=False)

    def _create_model_shareable(self) -> Shareable:
        """Create shareable from current model"""
        # Convert model parameters to numpy
        params = {name: param.cpu().numpy() for name, param in self.model.state_dict().items()}

        # Create FLModel with metadata
        fl_model = FLModel(
            params=params,
            meta={
                "client_name": self.client_name,
                "tasks": self.tasks,
                "dataname": self.dataname,
                "is_single_task": self.client_config.get('is_single_task', True),
                "client_id": self.client_config.get('client_id', 0),
            },
        )

        # Convert to shareable
        from nvflare.app_common.utils.fl_model_utils import FLModelUtils

        return FLModelUtils.to_shareable(fl_model)
