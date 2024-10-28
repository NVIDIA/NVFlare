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

import pytorch_lightning as pl
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder, PromptEncoderType
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

# configure logging at the root logging level
logging.getLogger().setLevel(logging.INFO)


class ServerPromptEncoder(PromptEncoder, FLComponent):
    def __init__(
        self,
        total_virtual_tokens: int = 10,
        hidden_size: int = 1024,
        taskname: str = "taskname",
        config_path: str = "config/megatron_gpt_prompt_learning_config.yaml",
        gpt_file_name: str = "megatron_gpt_345m.nemo",
        devices: int = 1,
    ):
        """
        Initializes the PromptEncoder module on the server.
        Args:
            total_virtual_tokens: the total number of virtual tokens
            hidden_size: hidden dimension
            taskname: prompt learning task name.
            config_path: NeMo model config file
            gpt_file_name: Pre-trained nemo model file.
            devices: number devices for cluster environment.
        """

        self.total_virtual_tokens = total_virtual_tokens
        self.hidden_size = hidden_size
        self.taskname = taskname
        self.config_path = config_path
        self.gpt_file_name = gpt_file_name
        self.devices = devices

        self.config = None

        FLComponent.__init__(self)

    def _init_environment(self):
        # setup cluster environment parameters
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(self.devices)

        strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
        plugins = [TorchElasticEnvironment()]
        trainer = pl.Trainer(plugins=plugins, strategy=strategy)

        # only needed to initialize the cluster environment
        _model = MegatronGPTPromptLearningModel(cfg=self.config.model, trainer=trainer)

    def _initialize(self, fl_ctx: FLContext):
        # get app root
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load model configuration to initialize training NeMo environment
        self.config = OmegaConf.load(os.path.join(app_root, self.config_path))
        self.config.trainer.devices = self.devices
        self.config.model.language_model_path = os.path.join(app_root, self.gpt_file_name)

        # Using defaults from `init_prompt_encoder` in `MegatronBasePromptLearningModel`
        _encoder_type = PromptEncoderType(self.config.model.p_tuning.get("encoder_type", "mlp").lower())

        if _encoder_type == PromptEncoderType.TPMLP:
            self._init_environment()

        PromptEncoder.__init__(
            self,
            encoder_type=_encoder_type,
            total_virtual_tokens=self.total_virtual_tokens,
            token_dim=self.hidden_size,
            hidden_size=self.config.model.p_tuning.get("encoder_hidden", self.hidden_size // 2),
            lstm_dropout=self.config.model.p_tuning.get("dropout", 0.0),
            num_layers=self.config.model.p_tuning.get("num_layers", 2),
            init_std=self.config.model.p_tuning.get("init_std", 0.023),
            taskname=self.taskname,
        )

        self.log_info(fl_ctx, f"Initialized prompt encoder type {_encoder_type}")

    def state_dict(self):
        _nemo_state_dict = PromptEncoder.state_dict(self)

        # Turn nested dict into single level dict supported by ModelPersistor and Aggregator
        state_dict = {}
        for encoder_key, prompt_state_dict in _nemo_state_dict.items():
            for k, v in prompt_state_dict.items():
                state_dict[f"{encoder_key}.{k}"] = v

        return state_dict

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)
