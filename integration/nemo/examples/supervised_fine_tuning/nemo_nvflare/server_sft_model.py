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

import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

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


class ServerSFTModel(torch.nn.Module, FLComponent):
    def __init__(
        self,
        config_path: str = "config/megatron_gpt_prompt_learning_config.yaml",
        base_model_file_path: str = "megatron_gpt_345m.nemo",
    ):
        """
        Initializes the NeMo model on the server.
        Args:
            config_path: NeMo model config file
            base_model_file_path: Pre-trained nemo model file
        """

        self.config_path = config_path
        self.base_model_file_path = base_model_file_path
        self.config = None
        FLComponent.__init__(self)
        torch.nn.Module.__init__(self)

    def _initialize(self, fl_ctx: FLContext):
        # get app root
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load model configuration to initialize training NeMo environment
        self.config = OmegaConf.load(os.path.join(app_root, self.config_path))
        self.config.model.restore_from_path = self.base_model_file_path
        # Trainer initialization, global model for persistence only, does not use GPU
        strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
        plugins = []
        trainer = Trainer(plugins=plugins, strategy=strategy, accelerator="cpu")
        # Load pretrained model
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(self.base_model_file_path):
            save_restore_connector.model_extracted_dir = self.base_model_file_path
        gpt_cfg = MegatronGPTSFTModel.restore_from(
            restore_path=self.config.model.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        self.model = load_from_nemo(MegatronGPTSFTModel, self.config, trainer, gpt_cfg, modify_config_fn=_modify_config)
        self.log_info(fl_ctx, "Initialized global model")

    def state_dict(self):
        return self.model.state_dict()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)
