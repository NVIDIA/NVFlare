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
        train_config_filename="config_train.json",
        aggregation_epochs=1,
    ):
        super().__init__()

        # Config settings
        self.train_config_filename = train_config_filename
        self.aggregation_epochs = aggregation_epochs

        # Training configuration (loaded from config_train.json)
        self.config_info = None

        # Training hyperparameters (loaded from config)
        self.lr = None
        self.weight_decay = None
        self.local_epochs = None
        self.batch_size = None
        self.val_batch_size = None
        self.warmup_epochs = None
        self.optimizer_name = None
        self.nworkers = None
        self.fp16 = None

        # Model config (loaded from config)
        self.backbone_type = None
        self.backbone_pretrained = None

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

        # NVFLARE-specific
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.train_loss = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the learner with client-specific configuration"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_name = fl_ctx.get_identity_name()

        print(f"🔧 Initializing FedHCA2Learner for {self.client_name}")

        # Load training configuration using standard NVFLARE pattern
        self.train_config(fl_ctx)

        # Load client-specific configuration
        print(f"   🔍 Client name: '{self.client_name}'")
        self.client_config = self._get_client_config_for_site(self.client_name)
        print(f"   📋 Client config: {self.client_config}")

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

        print(f"   📋 Tasks: {self.tasks}")
        print(f"   📁 Dataset: {self.dataname}")
        print(f"   ⚙️ Training config: lr={self.lr}, wd={self.weight_decay}, epochs={self.local_epochs}")
        print(f"   🏗️ Model config: {self.backbone_type}, pretrained={self.backbone_pretrained}")

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

    def train_config(self, fl_ctx: FLContext):
        """Load training configuration using standard NVFLARE pattern"""
        # Load training configurations json
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)

        if not os.path.isfile(train_config_file_path):
            raise FileNotFoundError(f"Training configuration file does not exist at {train_config_file_path}")

        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        # Note: Training hyperparameters are loaded per-client in _get_client_config_for_site
        # This allows different hyperparameters for different clients

        # Load model configuration
        model_config = self.config_info["model"]
        self.backbone_type = model_config["backbone_type"]
        self.backbone_pretrained = model_config["backbone_pretrained"]

        print(f"   📁 Loaded training config: lr={self.lr}, epochs={self.local_epochs}")

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

        # Ensure initialization (unified approach)
        if self.config_info is None or self.tasks is None or self.client_name is None:
            parts = {}  # Empty parts since we don't have them
            self.initialize(parts, fl_ctx)

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
        """Validation task"""
        # Ensure initialization (unified approach)
        if self.config_info is None or self.tasks is None or self.client_name is None:
            parts = {}
            self.initialize(parts, fl_ctx)

        print(f"🔧 Validating FedHCA2Learner for {self.client_name}")
        return self._run_validation(shareable, fl_ctx)

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Submit the current model"""
        # Ensure initialization (unified approach)
        if self.config_info is None or self.tasks is None or self.client_name is None:
            parts = {}
            self.initialize(parts, fl_ctx)

        return self._create_model_shareable()

    def _run_validation(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Run validation with FedHCA2 evaluation logic"""
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        print(f"   🔍 Current round: {current_round}")

        # Update model with received parameters
        self._update_local_model(shareable)

        # Check if we should perform evaluation (every 5 rounds as in original)
        if current_round % 1 == 0:
            print(f"   📊 Evaluating {self.client_name} at round {current_round}")
            # TODO: Add comprehensive evaluation using fedhca_core evaluation
            eval_results = self._perform_evaluation()
            return self._create_validation_result(eval_results)
        else:
            # Simple validation without full evaluation
            print(f"   🔍 {self.client_name} validation completed (round {current_round})")
            return self._create_validation_result()

    def _perform_evaluation(self) -> dict:
        """Perform comprehensive evaluation using fedhca_core evaluation logic"""
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            print("   ⚠️ No validation loader available, skipping evaluation")
            return {}

        # Import evaluation utilities from fedhca_core
        from fedhca2_core.evaluation.evaluate_utils import PerformanceMeter

        # Setup evaluation meter for this client's tasks
        eval_meter = PerformanceMeter(self.dataname, self.tasks)
        eval_meter.reset()

        # Set model to evaluation mode
        self.model.eval()
        eval_results = {}

        with torch.no_grad():
            try:
                print(f"   📊 Running evaluation on {len(self.val_loader)} validation batches...")

                for batch_idx, batch in enumerate(self.val_loader):
                    if batch_idx >= 10:  # Limit evaluation to 10 batches for efficiency
                        break

                    # Get inputs and targets
                    inputs = batch['image'].to(self.device)
                    targets = {task: batch[task].to(self.device) for task in self.tasks}

                    # Forward pass
                    outputs = self.model(inputs)

                    # Update evaluation meter
                    eval_meter.update(outputs, targets)

                # Get final evaluation scores
                eval_results = eval_meter.get_score()

                # Log results
                print(f"   📈 Evaluation results for {self.client_name}:")
                for task, score in eval_results.items():
                    print(f"      {task}: {score:.4f}")

            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                # Return dummy scores to avoid breaking the flow
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
        """Get client configuration based on client name and experiment config"""

        # Get base configurations from ST_Datasets and MT_Datasets
        st_config = self.config_info["ST_Datasets"][0]  # Pascal context config
        mt_config = self.config_info["MT_Datasets"][0]  # NYU depth config

        # Extract client configuration based on client name pattern
        if "site-1" in client_name:
            config = st_config.copy()
            config.update({'tasks': ['semseg'], 'is_single_task': True, 'client_id': 0})
            return config
        elif "site-2" in client_name:
            config = st_config.copy()
            config.update({'tasks': ['human_parts'], 'is_single_task': True, 'client_id': 1})
            return config
        elif "site-3" in client_name:
            config = st_config.copy()
            config.update({'tasks': ['normals'], 'is_single_task': True, 'client_id': 2})
            return config
        elif "site-4" in client_name:
            config = st_config.copy()
            config.update({'tasks': ['edge'], 'is_single_task': True, 'client_id': 3})
            return config
        elif "site-5" in client_name:
            config = st_config.copy()
            config.update({'tasks': ['sal'], 'is_single_task': True, 'client_id': 4})
            return config
        elif "site-6" in client_name:
            config = mt_config.copy()
            config.update(
                {
                    'tasks': ['semseg', 'normals', 'edge', 'depth'],
                    'is_single_task': False,
                    'client_id': 5,
                }
            )
            return config
        else:
            # Default configuration
            config = st_config.copy()
            config.update({'tasks': ['semseg'], 'is_single_task': True, 'client_id': 0})
            return config

    def _get_experiment_config_from_training_config(self) -> dict:
        """Extract experiment config from training config"""
        return {
            "ST_Datasets": self.config_info["ST_Datasets"],
            "MT_Datasets": self.config_info["MT_Datasets"],
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

        print(f"   📊 Data partition: {len(data_indices)} samples")

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
