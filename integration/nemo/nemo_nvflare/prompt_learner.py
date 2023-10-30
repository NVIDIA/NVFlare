# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os

import nemo
import numpy as np
import pytorch_lightning as pl
import torch
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_opt.lightning.callbacks import RestoreState
from nvflare.fuel.utils.network_utils import get_open_ports

from .constants import NemoDataKind
from .fed_megatron_gpt_prompt_learning_model import FedMegatronGPTPromptLearningModel
from .utils import compute_model_diff, load_weights

print("NEMO version", nemo.__version__)
# configure logging at the root logging level
logging.getLogger().setLevel(logging.INFO)


def set_datafile_paths(files, app_root):
    new_files = []
    for f in files:
        f = os.path.join(app_root, f)
        if not os.path.isfile(f):
            raise ValueError(f"No such file {f}!")
        new_files.append(f)
    return new_files


class PromptLearner(Learner):
    def __init__(
        self,
        config_path: str = None,
        train_ds_files: str = "financial_phrase_bank_train.jsonl",
        val_ds_files: str = "financial_phrase_bank_val.jsonl",
        task_templates_file: str = None,
        gpt_file_name: str = "megatron_gpt_345m.nemo",
        nemo_path: str = "multitask_p_tuned_gpt.nemo",
        exp_name: str = "prompt_learning",
        existing_tasks: str = None,
        new_tasks: str = "taskname",
        aggregation_epochs: int = 1,
        master_addr: str = "localhost",
        master_port: int = None,
        devices: int = 1,
        virtual_prompt_style=VirtualPromptStyle.P_TUNING,
        key_metric: str = "global_model_val_loss",
        negate_key_metric: bool = True,
    ):
        """Support prompt learning with NeMo

        Args:
            config_path: NeMo model config file
            train_ds_files: Training dataset files.
            val_ds_files: Validation dataset files.
            task_templates_file: Task template file
            gpt_file_name: Pre-trained nemo model file.
            nemo_path: Where to store the locally p-tuned model.
            exp_name: Name of current experiment.
            existing_tasks: Existing task names.
            new_tasks: New task name.
            aggregation_epochs: the number of training epochs for a round.
            master_addr: Master node (rank 0)'s address, should be either the IP address or the hostname of node 0.
            master_port: Master node (rank 0)'s free port.
            devices: number devices for cluster environment.
            virtual_prompt_style: Style of prompt learning method. Defaults to p-tuning (`VirtualPromptStyle.P_TUNING`).
            key_metric: Key metric for global model selection. Defaults to `"global_model_val_loss"`.
            negate_key_metric: Whether to invert the key metric. Should be used if key metric is a loss. Default to `True`.

        Returns:
            a Shareable with the updated local model after running `train()`
            or the validation metric when calling `validate()`.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

        self.config_path = config_path
        self.train_ds_files = train_ds_files
        self.val_ds_files = val_ds_files
        self.task_templates_file = task_templates_file
        self.gpt_file_name = gpt_file_name
        self.nemo_path = nemo_path
        self.exp_name = exp_name
        self.existing_tasks = existing_tasks
        self.new_tasks = new_tasks
        self.aggregation_epochs = aggregation_epochs
        self.master_addr = master_addr
        self.master_port = master_port
        self.devices = devices
        self.virtual_prompt_style = virtual_prompt_style
        self.key_metric = key_metric
        self.negate_key_metric = negate_key_metric

        self.app_root = None
        self.client_id = None
        self.config = None
        self.trainer = None
        self.model = None
        self.is_configured = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_configs(self, configs):
        if not isinstance(configs, dict):
            raise ValueError(f"Exptected configs to be of type dict but received type {type(configs)}")

        # Received primitive dicts from server; convert back to OmegaConf
        if NemoDataKind.NEMO_CONFIG in configs:
            self.config = OmegaConf.create(configs[NemoDataKind.NEMO_CONFIG])
        else:
            raise ValueError(f"Received configs did not contain nemo configs! Received keys: {list(configs.keys())}")

        if NemoDataKind.TASK_TEMPLATES in configs:
            self.config.model.task_templates = OmegaConf.create(configs[NemoDataKind.TASK_TEMPLATES])
        else:
            raise ValueError(f"Received configs did not contain task templates! Received keys: {list(configs.keys())}")

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Build model, training & validation data sets
        """

        # when the run starts, this is where the actual settings get initialized for trainer
        self.log_info(fl_ctx, "Initializing the Learner...")
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.client_id = fl_ctx.get_identity_name()

        if self.devices > 1:
            # distributed environment is set by PTMultiProcessExecutor
            if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
                raise ValueError(
                    f"Distributed environment not set up correctly for {self.devices} devices. "
                    f"Did you use `PTMultiProcessExecutor`?"
                )
        else:
            # Setup cluster environment parameters
            # use torch elastic cluster environment so `create_process_externally` is True
            # the launcher is set to None. It will not try to spawn new processes.
            # It won't create the misconfiguration error because of the `interactive session`
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port) if self.master_port else str(get_open_ports(1)[0])

            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
        self.log_info(
            fl_ctx,
            f"Running with distributed environment: LOCAL_RANK: {os.environ['LOCAL_RANK']}, "
            f"RANK: {os.environ['RANK']}, WORLD_SIZE {os.environ['WORLD_SIZE']}, "
            f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, and MASTER_PORT: {os.environ['MASTER_PORT']}",
        )

    def _check_new_tasks(self):
        template_tasks = [t.get("taskname") for t in self.config.model.task_templates]
        missing_tasks = []
        for _new_task in self.new_tasks:
            if _new_task not in template_tasks:
                missing_tasks.append(_new_task)
        if any(missing_tasks):
            raise ValueError(f"New tasks {missing_tasks} not specified in task templates {template_tasks}!")

    def _configure(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Configuring the Learner...")
        # Load model configuration
        if self.config_path is not None:
            if self.config is not None:
                self.log_warning(fl_ctx, "Attempting to overwrite config received from server...")
            self.config_path = os.path.join(self.app_root, self.config_path)
            self.log_info(fl_ctx, f"Load model configuration from {self.config_path}")
            self.config = OmegaConf.load(self.config_path)

        if self.config is None:
            raise ValueError("No configuration was received or loaded!")

        # Load task templates
        if self.task_templates_file is not None:
            if self.config.model.task_templates is not None:
                self.log_warning(fl_ctx, "Attempting to overwrite task templates received from server...")
            self.task_templates_file = os.path.join(self.app_root, self.task_templates_file)
            self.log_info(fl_ctx, f"Load task templates from {self.task_templates_file}")
            self.config.model.task_templates = OmegaConf.load(self.task_templates_file)
        if self.config.model.task_templates is None:
            raise ValueError("No task templates were received or loaded!")

        # Specify existing tasks
        if not self.existing_tasks:
            self.config.model.existing_tasks = []
        else:
            self.config.model.existing_tasks = self.existing_tasks

        # Set tasks to learn
        if not isinstance(self.new_tasks, list):
            self.new_tasks = [self.new_tasks]
        self.config.model.new_tasks = self.new_tasks

        # check if all new tasks are in the task templates
        self._check_new_tasks()

        # Configure training sets
        if not isinstance(self.train_ds_files, list):
            self.train_ds_files = [self.train_ds_files]
        if not isinstance(self.val_ds_files, list):
            self.val_ds_files = [self.val_ds_files]
        self.config.model.data.train_ds = set_datafile_paths(self.train_ds_files, self.app_root)
        self.config.model.data.validation_ds = set_datafile_paths(self.val_ds_files, self.app_root)

        # Set GPT model path on prompt learning config
        self.config.model.language_model_path = self.gpt_file_name

        # We can also set where we want the final prompt tuned model to be saved by setting `model.nemo_path`.
        self.config.model.nemo_path = os.path.join(self.app_root, self.nemo_path)

        # Setting P-Tuning Specific Params
        self.config.model.virtual_prompt_style = self.virtual_prompt_style

        # Configure in yaml file
        self.log_info(
            fl_ctx,
            f"Training with global_batch_size {self.config.model.global_batch_size}"
            f" and micro_batch_size {self.config.model.micro_batch_size}",
        )

        # for PyTorch Native AMP set precision=16 (use value from config yaml)
        self.config.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.config.model.tensor_model_parallel_size = self.devices

        self.config.trainer.devices = self.devices
        self.config.trainer.max_epochs = -1  # Needed to continue fit() in next round

        strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
        plugins = [TorchElasticEnvironment()]

        # Add TensorBoard logger
        self.config.trainer.logger = True
        self.config.trainer.default_root_dir = self.app_root

        self.trainer = pl.Trainer(plugins=plugins, strategy=strategy, callbacks=[RestoreState()], **self.config.trainer)
        self.config.model.precision = self.config.trainer.precision

        # Set name of the experiment
        self.config.name = self.exp_name

        self.log_info(fl_ctx, f"Model config - {OmegaConf.to_yaml(self.config.model)}")
        self.log_info(fl_ctx, f"Trainer config - {OmegaConf.to_yaml(self.config.trainer)}")

        # The only thing left to do is load up the model and begin p-tuning!
        self.model = FedMegatronGPTPromptLearningModel(cfg=self.config.model, trainer=self.trainer)
        self.model.init_prompt_encoder()

        self.is_configured = True
        self.log_info(
            fl_ctx, f"Initialized model {type(self.model)} and prompt encoder {type(self.model.prompt_encoder)}"
        )

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        if not self.is_configured:
            self._configure(fl_ctx)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        if current_round > 0:
            self.trainer.num_sanity_val_steps = 0  # Turn off sanity validation steps in 2nd round of FL
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        if not self.model.prompt_encoder:
            raise ValueError("Prompt encoder is not available!")

        n_loaded = load_weights(self.model, global_weights, device=self.device)
        self.log_info(fl_ctx, f"Loaded {n_loaded} of {len(global_weights)} weights")

        self.log_info(fl_ctx, f"Start training in round {current_round}")
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch + self.aggregation_epochs

        self.model.log_global = False
        self.trainer.fit(self.model)

        model_diff = compute_model_diff(self.model, global_weights)
        self.log_info(
            fl_ctx, f"Computed {len(model_diff)} weight differences for global model of length {len(global_weights)}"
        )

        # Get local steps from data loader
        epoch_len = len(self.model._train_dl)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        if not self.is_configured:
            self._configure(fl_ctx)

        if not self.model.prompt_encoder:
            raise ValueError("Prompt encoder is not available!")

        n_loaded = load_weights(self.model, global_weights, device=self.device)
        self.log_info(fl_ctx, f"Loaded {n_loaded} of {len(global_weights)} weights")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            self.model.log_global = True  # enable logging the global metric
            global_metrics = self.trainer.validate(self.model)

            metric = global_metrics[0].get(self.key_metric, np.nan)

            self.log_info(fl_ctx, f"Global_model {self.key_metric}: {metric}")

            if self.negate_key_metric:
                metric = -1.0 * metric

            # use negative validation loss as validation metric
            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: metric}, meta={}).to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
