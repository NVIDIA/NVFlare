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

import torch
from apex.transformer import parallel_state
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer


class FedMegatronGPTPromptLearningModel(MegatronGPTPromptLearningModel):
    """
    Federated Learning Model class for prompt-tuning or p-tuning a pretrained Megatron GPT model.
    Adapted from https://github.com/NVIDIA/NeMo/blob/v1.17.0/nemo/collections/nlp/models/language_modeling/megatron_gpt_prompt_learning_model.py

    Prompt Tuning initializes virtual prompt embeddings directly from a copy of
    certain token embeddings from the pretrained GPT model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in
    initialization are specified by the user in the config file. The model can
    be prompt-tuned for multiple tasks at once. virtual prompts are stored in a
    prompt table and can be added or deleted without disrupting virtual prompts
    for other tasks.

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After p-tuning
    is complete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a
    new virtual prompt via p-tuning, they do not need to retrain on all previous
    tasks. This gives p-tuning the same task flexibility as prompt-tuning.

    Args:
        cfg: NeMo model configuration file
        trainer: PyTorch Lighting Trainer
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        self.is_initialized = False
        self.log_global = False

    def setup(self, stage=None):
        """Customize the prompt encoder setup"""

        if stage == "predict" and self.first_stage_of_pipeline():
            self.freeze_existing_word_embeddings()
            return

        self.setup_test_data()
        if stage == "test":
            return

        if self.first_stage_of_pipeline():
            # Differently from setup() in the super class,
            # here we don't initialize the prompt encoder as that
            # would overwrite the global weights from the server
            self.freeze_existing_word_embeddings()

        # Only initialize the datasets once
        if not self.is_initialized:
            self.setup_training_data()
            self.setup_validation_data()
            self.is_initialized = True

    def validation_epoch_end(self, outputs):
        """Use same logic as in `MegatronGPTPromptLearningModel` but change the logging tag name"""

        if self.log_global:  # log the global model
            log_name = "global_model_val_loss"
        else:
            log_name = "val_loss"

        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log(log_name, averaged_loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        logging.info(f"{log_name}: {averaged_loss}")

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
