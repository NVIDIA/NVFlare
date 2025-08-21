"""
FedHCA2 NVFLARE Learner - Clean wrapper around original FedHCA2 training logic
"""

import copy
import json
import os

import torch
import torch.optim as optim

# Import NVFLARE-specific data utilities
from data_utils.data_loader import get_client_data_partition, get_dataloader, get_dataset
from fedhca2_core.datasets.custom_transforms import get_transformations
from fedhca2_core.datasets.utils.configs import TEST_SCALE, TRAIN_SCALE
from fedhca2_core.losses import get_criterion

# Import original FedHCA2 components
from fedhca2_core.models.build_models import build_model
from fedhca2_core.train_utils import eval_metric as fedhca2_eval_metric
from fedhca2_core.train_utils import local_train as fedhca2_local_train
from fedhca2_core.utils import RunningMeter, get_output, set_seed, to_cuda
from timm.scheduler.cosine_lr import CosineLRScheduler

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants

# Note: local_train is now imported from fedhca2_core.train_utils as fedhca2_local_train


class FedHCA2Learner(Executor):
    """
    Clean NVFLARE wrapper around original FedHCA2 client logic
    """

    def __init__(
        self,
        lr=0.0001,
        weight_decay=0.0001,
        local_epochs=1,
        batch_size=4,
        warmup_epochs=5,
        fp16=True,
        backbone_type="swin-t",
        backbone_pretrained=True,
    ):
        super().__init__()

        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.fp16 = fp16
        self.backbone_type = backbone_type
        self.backbone_pretrained = backbone_pretrained

        # Will be initialized in initialize()
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.train_loss = None

        # Client-specific configuration
        self.client_name = None
        self.tasks = None
        self.dataname = None
        self.client_config = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the learner with client-specific configuration"""
        # Note: Executor may not have an initialize method, so we skip super() call

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_name = fl_ctx.get_identity_name()

        print(f"🔧 Initializing FedHCA2Learner for {self.client_name}")
        print(f"   🔧 Initialize method called successfully!")

        # Load client configuration from experiment config
        print(f"   🔍 Client name: '{self.client_name}'")
        self.client_config = self._load_client_config(fl_ctx)
        print(f"   📋 Client config: {self.client_config}")

        if self.client_config is None:
            raise ValueError(f"Failed to load configuration for client: {self.client_name}")

        self.tasks = self.client_config.get('tasks')
        self.dataname = self.client_config.get('dataname')

        if self.tasks is None:
            raise ValueError(f"Tasks not found in config for client: {self.client_name}")
        if self.dataname is None:
            raise ValueError(f"Dataname not found in config for client: {self.client_name}")

        print(f"   📋 Tasks: {self.tasks}")
        print(f"   📁 Dataset: {self.dataname}")

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
        max_epochs = 100 * self.local_epochs  # Approximate total
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

        # Initialize loss meters
        self.train_loss = {task: RunningMeter() for task in self.tasks}

        print(f"   ✅ {self.client_name} initialized successfully")

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
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        # Fallback initialization if initialize was not called
        if self.tasks is None or self.client_name is None:
            print("⚠️ WARNING: initialize() was not called! Performing fallback initialization...")
            try:
                parts = {}  # Empty parts since we don't have them
                self.initialize(parts, fl_ctx)
            except Exception as e:
                print(f"❌ Fallback initialization failed: {e}")
                # Manual initialization as last resort
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client_name = fl_ctx.get_identity_name()
                self.client_config = self._load_client_config(fl_ctx)
                self.tasks = self.client_config.get('tasks', ['semseg'])
                self.dataname = self.client_config.get('dataname', 'pascalcontext')

                print(f"   🆘 Manual initialization: {self.client_name}, tasks: {self.tasks}")

                # Build model
                if self.model is None:
                    print(f"      🏗️ Building model...")
                    self.model = build_model(
                        tasks=self.tasks,
                        dataname=self.dataname,
                        backbone_type=self.backbone_type,
                        backbone_pretrained=self.backbone_pretrained,
                    ).to(self.device)

                # Build optimizer
                if self.optimizer is None:
                    print(f"      ⚙️ Building optimizer...")
                    self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                # Build scheduler
                if self.scheduler is None:
                    print(f"      📅 Building scheduler...")
                    max_epochs = 100 * self.local_epochs  # Approximate total
                    self.scheduler = CosineLRScheduler(
                        optimizer=self.optimizer,
                        t_initial=max_epochs - self.warmup_epochs,
                        lr_min=1.25e-6,
                        warmup_t=self.warmup_epochs,
                        warmup_lr_init=1.25e-7,
                        warmup_prefix=True,
                    )

                # Build criterion
                if self.criterion is None:
                    print(f"      🎯 Building criterion...")
                    self.criterion = get_criterion(self.dataname, self.tasks).to(self.device)

                # Build scaler
                if self.scaler is None:
                    print(f"      🚀 Building scaler...")
                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

                # Build data loaders (real implementation)
                if self.train_loader is None:
                    print(f"      📊 Building data loaders...")
                    self._setup_data_loaders(fl_ctx)

                # Initialize training loss meters
                if self.train_loss is None:
                    self.train_loss = {task: RunningMeter() for task in self.tasks}

                print(f"   ✅ Manual initialization completed successfully")

        print(f"\n🔥 {self.client_name} - Round {current_round} Training")
        print(f"   🎯 Tasks: {self.tasks}")
        print(f"   🔄 Local Epochs: {self.local_epochs}")

        # Update model with received parameters
        self._update_local_model(shareable)

        # Reset loss meters
        for task in self.tasks:
            self.train_loss[task].reset()

        # Original FedHCA2 local training
        fedhca2_local_train(
            idx=0,  # Client index (can use 0 for NVFLARE since client name is handled separately)
            cr=current_round if current_round is not None else 0,  # Current round
            local_epochs=self.local_epochs,
            tasks=self.tasks,
            train_dl=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            scaler=self.scaler,
            train_loss=self.train_loss,
            local_rank=0,  # For simplicity, assuming single GPU per client
            fp16=self.fp16,
        )

        print(f"   📊 Training completed:")
        for task, loss_meter in self.train_loss.items():
            print(f"      {task}: {loss_meter.avg:.4f}")

        # Return model parameters
        return self._create_model_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Validate the model"""
        self._update_local_model(shareable)

        # Simple validation - just return success
        # In real implementation, would compute validation metrics
        print(f"   🔍 {self.client_name} validation completed")

        return self._create_model_shareable()

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Submit the current model"""
        return self._create_model_shareable()

    def _load_client_config(self, fl_ctx):
        """Load client-specific configuration"""
        # Extract client configuration based on client name
        # For now, use hardcoded config based on client name pattern

        if "site-1" in self.client_name:
            return {'dataname': 'pascalcontext', 'tasks': ['semseg'], 'is_single_task': True, 'client_id': 0}
        elif "site-2" in self.client_name:
            return {'dataname': 'pascalcontext', 'tasks': ['human_parts'], 'is_single_task': True, 'client_id': 1}
        elif "site-3" in self.client_name:
            return {'dataname': 'pascalcontext', 'tasks': ['normals'], 'is_single_task': True, 'client_id': 2}
        elif "site-4" in self.client_name:
            return {'dataname': 'pascalcontext', 'tasks': ['edge'], 'is_single_task': True, 'client_id': 3}
        elif "site-5" in self.client_name:
            return {'dataname': 'pascalcontext', 'tasks': ['sal'], 'is_single_task': True, 'client_id': 4}
        elif "site-6" in self.client_name:
            return {
                'dataname': 'nyud',
                'tasks': ['semseg', 'normals', 'edge', 'depth'],
                'is_single_task': False,
                'client_id': 5,
            }
        else:
            # Default configuration
            return {'dataname': 'pascalcontext', 'tasks': ['semseg'], 'is_single_task': True, 'client_id': 0}

    def _setup_data_loaders(self, fl_ctx):
        """Setup data loaders with client-specific partitioning"""
        # Get transforms
        train_transform = get_transformations(TRAIN_SCALE[self.dataname], train=True)
        val_transform = get_transformations(TEST_SCALE[self.dataname], train=False)

        # Get experiment config for data partitioning
        exp_config = self._load_experiment_config(fl_ctx)

        # Get client-specific data partition
        data_indices = get_client_data_partition(self.client_config['client_id'], self.client_config, exp_config)

        print(f"   📊 Data partition: {len(data_indices)} samples")

        # Create datasets
        train_dataset = get_dataset(
            dataname=self.dataname, tasks=self.tasks, train=True, dataidxs=data_indices, transform=train_transform
        )

        val_dataset = get_dataset(dataname=self.dataname, tasks=self.tasks, train=False, transform=val_transform)

        # Create data loaders
        train_configs = {
            'tr_batch': self.batch_size,
            'val_batch': self.batch_size,
            'nworkers': 0,  # Set to 0 for compatibility
        }

        self.train_loader = get_dataloader(train=True, configs=train_configs, dataset=train_dataset)

        self.val_loader = get_dataloader(train=False, configs=train_configs, dataset=val_dataset)



    def _load_experiment_config(self, fl_ctx):
        """Load experiment configuration"""
        # For now, return hardcoded configuration
        # In real implementation, would load from config file
        return {
            "ST_Datasets": [
                {
                    "dataname": "pascalcontext",
                    "task_dict": {"semseg": 1, "human_parts": 1, "normals": 1, "edge": 1, "sal": 1},
                }
            ],
            "MT_Datasets": [
                {"dataname": "nyud", "client_num": 1, "task_dict": {"semseg": 1, "normals": 1, "edge": 1, "depth": 1}}
            ],
        }

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
                # Load personalized model
                params = personalized_models[self.client_name]
                state_dict = {k: torch.tensor(v) for k, v in params.items()}
                self.model.load_state_dict(state_dict, strict=False)
                print(f"   🎯 Loaded personalized model for {self.client_name}")
                return

        # Load global model
        state_dict = {k: torch.tensor(v) for k, v in fl_model.params.items()}
        self.model.load_state_dict(state_dict, strict=False)
        print(f"   📊 Loaded global model")

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
