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
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

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
from .utils_sft import compute_model_diff, load_weights

print("NEMO version", nemo.__version__)
# configure logging at the root logging level
logging.getLogger().setLevel(logging.INFO)


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get("hidden_dropout", 0.0)
        gpt_cfg.attention_dropout = cfg.model.get("attention_dropout", 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


def load_from_nemo(cls, cfg, trainer, gpt_cfg, modify_config_fn):
    gpt_cfg = modify_config_fn(gpt_cfg, cfg, add_cfg_to_tree=False)
    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
    model = cls.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=gpt_cfg,
        save_restore_connector=save_restore_connector,
    )
    return model


class SFTLearner(Learner):
    def __init__(
        self,
        config_path: str = None,
        train_ds_files: str = "financial_phrase_bank_train.jsonl",
        validation_ds_files: str = "financial_phrase_bank_val.jsonl",
        base_model_file_path: str = "megatron_gpt_345m.nemo",
        sft_model_file_path: str = "megatron_gpt_345m_sft.nemo",
        aggregation_epochs: int = 1,
        master_addr: str = "localhost",
        master_port: int = None,
        devices: int = 1,
        key_metric: str = "val_loss",
    ):
        """Support SFT (Supervised Fine-Tuning) learning with NeMo

        Args:
            config_path: NeMo model config file
            train_ds_files: Training dataset files.
            validation_ds_files: Validation dataset files.
            base_model_file_path: Pre-trained nemo model file.
            sft_model_file_path: Where to store the local SFT model.
            aggregation_epochs: the number of training epochs for a round.
            master_addr: Master node (rank 0)'s address, should be either the IP address or the hostname of node 0.
            master_port: Master node (rank 0)'s free port.
            devices: number devices for cluster environment.
            key_metric: Key metric for global model selection. Defaults to `"global_model_validation_loss"`.

        Returns:
            a Shareable with the updated local model after running `train()`
            or the validation metric when calling `validate()`.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

        self.config_path = config_path
        self.train_ds_files = train_ds_files
        self.validation_ds_files = validation_ds_files
        self.base_model_file_path = base_model_file_path
        self.sft_model_file_path = sft_model_file_path
        self.aggregation_epochs = aggregation_epochs
        self.master_addr = master_addr
        self.master_port = master_port
        self.devices = devices
        self.key_metric = key_metric

        self.app_root = None
        self.client_id = None
        self.config = None
        self.trainer = None
        self.model = None
        self.is_configured = False
        self.steps_per_round = None
        self.scaler = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_configs(self, configs):
        if not isinstance(configs, dict):
            raise ValueError(f"Exptected configs to be of type dict but received type {type(configs)}")

        # Received primitive dicts from server; convert back to OmegaConf
        if NemoDataKind.NEMO_CONFIG in configs:
            self.config = OmegaConf.create(configs[NemoDataKind.NEMO_CONFIG])
        else:
            raise ValueError(f"Received configs did not contain nemo configs! Received keys: {list(configs.keys())}")

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
            if "MASTER_ADDR" not in os.environment or "MASTER_PORT" not in os.environment:
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

        # Configure training sets
        if not os.path.isfile(self.train_ds_files):
            raise ValueError(f"Training data file not found: {self.train_ds_files}!")
        if not os.path.isfile(self.validation_ds_files):
            raise ValueError(f"Validation data file not found: {self.validation_ds_files}!")
        self.config.model.data.train_ds.file_names = [self.train_ds_files]
        self.config.model.data.validation_ds.file_names = [self.validation_ds_files]

        # Set the base model path for further SFT
        self.config.model.restore_from_path = self.base_model_file_path

        # We can also set where we want the final SFT tuned model to be saved by setting `model.nemo_path`.
        self.config.model.nemo_path = os.path.join(self.app_root, self.sft_model_file_path)

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
        # self.config.trainer.max_epochs = -1  # Needed to continue fit() in next round

        megatron_amp_o2 = self.config.model.get("megatron_amp_O2", False)
        with_distributed_adam = self.config.model.optim.get("name", "fused_adam") == "distributed_fused_adam"

        plugins = []
        strategy = NLPDDPStrategy(
            no_ddp_communication_hook=True,
            gradient_as_bucket_view=self.config.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )

        if self.config.trainer.precision in [16, "bf16"]:
            if self.config.trainer.precision == 16:
                self.scaler = GradScaler(
                    init_scale=self.config.model.get("native_amp_init_scale", 2**32),
                    growth_interval=self.config.model.get("native_amp_growth_interval", 1000),
                    hysteresis=self.config.model.get("hysteresis", 2),
                )
            if megatron_amp_o2 and not with_distributed_adam:
                plugins.append(
                    MegatronHalfPrecisionPlugin(
                        precision=self.config.trainer.precision, device="cuda", scaler=self.scaler
                    )
                )
            else:
                plugins.append(
                    PipelineMixedPrecisionPlugin(
                        precision=self.config.trainer.precision, device="cuda", scaler=self.scaler
                    )
                )

        # Add TensorBoard logger
        self.config.exp_manager.explicit_log_dir = self.app_root

        self.trainer = Trainer(plugins=plugins, strategy=strategy, callbacks=[RestoreState()], **self.config.trainer)
        exp_manager(self.trainer, self.config.exp_manager)
        self.log_info(fl_ctx, f"Model config - {OmegaConf.to_yaml(self.config.model)}")
        self.log_info(fl_ctx, f"Trainer config - {OmegaConf.to_yaml(self.config.trainer)}")

        # Load pretrained model from nemo
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(self.config.model.restore_from_path):
            save_restore_connector.model_extracted_dir = self.config.model.restore_from_path
        gpt_cfg = MegatronGPTSFTModel.restore_from(
            restore_path=self.config.model.restore_from_path,
            trainer=self.trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        self.model = load_from_nemo(
            MegatronGPTSFTModel, self.config, self.trainer, gpt_cfg, modify_config_fn=_modify_config
        )

        self.is_configured = True
        self.log_info(fl_ctx, f"Initialized model {type(self.model)}")

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

        if current_round == 0:
            self.steps_per_round = self.trainer.fit_loop.max_steps
        if current_round > 0:
            self.trainer.num_sanity_validation_steps = 0  # Turn off sanity validation steps in 2nd round of FL
            self.trainer.fit_loop.max_epochs += self.aggregation_epochs
            self.trainer.fit_loop.max_steps += self.steps_per_round

        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        n_loaded = load_weights(self.model, global_weights, device=self.device)
        self.log_info(fl_ctx, f"Loaded {n_loaded} of {len(global_weights)} weights")

        self.log_info(fl_ctx, f"Start training in round {current_round}")

        self.log_info(fl_ctx, f"Current max_steps {self.trainer.fit_loop.max_steps}")
        self.log_info(fl_ctx, f"Current max_epochs {self.trainer.fit_loop.max_epochs}")

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

        n_loaded = load_weights(self.model, global_weights, device=self.device)
        self.log_info(fl_ctx, f"Loaded {n_loaded} of {len(global_weights)} weights")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            self.model.log_global = True  # enable logging the global metric
            global_metrics = self.trainer.validate(self.model)
            metric = global_metrics[0].get(self.key_metric)
            self.log_info(fl_ctx, f"Global_model {self.key_metric}: {metric}")

            # use validation loss as validation metric
            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: metric}, meta={}).to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
