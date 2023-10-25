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
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

# configure logging at the root logging level
logging.getLogger().setLevel(logging.INFO)


class PEFTmodel(torch.nn.Module, FLComponent):
    def __init__(
        self,
        config_path: str = "custom/megatron_gpt_peft_tuning_config.yaml",
        restore_from_path: str = "/home/hroth/Code2/nvflare/nemo_peft_example/integration/nemo/examples/peft/megatron_gpt_345m.nemo"
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

        self.config_path = config_path
        self.restore_from_path = restore_from_path

        torch.nn.Module.__init__(self)
        FLComponent.__init__(self)

    def _initialize(self, fl_ctx: FLContext):
        # get app root
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load model configuration
        cfg = OmegaConf.load(os.path.join(app_root, self.config_path))

        # Build trainer
        trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

        # Set restore from path with pre-trained model
        cfg.model.restore_from_path = os.path.join(app_root, self.restore_from_path)

        # Set some dummy data files (which will not be used)
        cfg.model.data.train_ds.file_names = ["dummy.jsonl"]
        cfg.model.data.validation_ds.file_names = ["dummy.jsonl"]

        model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        self.model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

        logging.info("Adding adapter weights to the model for PEFT")
        self.model.add_adapter(peft_cfg_cls(model_cfg))

    def state_dict(self):
        _peft_state_dict = self.model.get_peft_state_dict()

        print("###### SERVER _peft_state_dict", _peft_state_dict.keys())

        return _peft_state_dict

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)
