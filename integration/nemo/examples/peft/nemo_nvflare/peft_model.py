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
        restore_from_path: str = "/models/megatron_gpt_345m.nemo",
        peft_restore_from_path: str = None,
    ):
        """
        Initializes the PEFT model or full model on the server.
        Args:
            config_path: NeMo model config file
            restore_from_path: Pre-trained NeMo model file.
            peft_restore_from_path: Pre-trained peft model file.
        """

        self.config_path = config_path
        self.restore_from_path = restore_from_path
        self.peft_restore_from_path = peft_restore_from_path

        self.use_sft = False

        torch.nn.Module.__init__(self)
        FLComponent.__init__(self)

    def _initialize(self, fl_ctx: FLContext):
        # importing nemo can take some time. Moving to initialize during START_RUN.
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
        from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
        from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
        from omegaconf import OmegaConf

        # get app root
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load model configuration
        cfg = OmegaConf.load(os.path.join(app_root, self.config_path))

        # Build trainer
        trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

        # Set restore from paths with pre-trained model(s)
        cfg.model.restore_from_path = os.path.join(app_root, self.restore_from_path)
        if self.peft_restore_from_path is not None:
            cfg.model.peft.restore_from_path = os.path.join(app_root, self.peft_restore_from_path)

        # Set some dummy data file names (which will not be used and do not need to exist)
        cfg.model.data.train_ds.file_names = ["dummy.jsonl"]
        cfg.model.data.validation_ds.file_names = ["dummy.jsonl"]

        model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        self.model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

        if cfg.model.peft.restore_from_path is not None:
            # initialize peft weights from a checkpoint instead of randomly
            # This is not the same as resume training because optimizer states are not restored.
            logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
            self.model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
        elif peft_cfg_cls is not None:
            logging.info("Adding adapter weights to the model for PEFT")
            self.model.add_adapter(peft_cfg_cls(model_cfg))
        else:
            self.use_sft = True
            logging.info(f"Running full finetuning since no peft scheme is given.\n{self.model.summarize()}")

    def state_dict(self):
        if self.use_sft:  # return the full model state dict
            state_dict = self.model.state_dict()
        else:  # only return the trainable peft parameters
            state_dict = self.model.get_peft_state_dict()

        # Fill any tensors that are not initialized.
        # This is sometimes needed for buffer tensors such as 'inference_table'
        for k, v in state_dict.items():
            if torch.any(torch.isnan(v)):
                state_dict[k] = state_dict[k].fill_(0.0)

        return state_dict

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)
