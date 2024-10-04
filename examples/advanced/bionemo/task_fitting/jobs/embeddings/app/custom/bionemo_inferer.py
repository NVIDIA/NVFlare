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


import os
import pickle
import uuid

import torch
from bionemo.data.mapped_dataset import FilteredMappedDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.utils import expand_dataset_paths
from bionemo_constants import BioNeMoConstants
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVFieldsMemmapDataset
from nemo.utils.distributed import gather_objects
from nemo.utils.model_utils import import_class_by_path
from omegaconf.omegaconf import OmegaConf

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.security.logging import secure_format_exception


class BioNeMoInferer(Executor):
    """Runs inference over all models. Supports extracting embeddings, and hiddens.

    NOTE: If out of memory (OOM) error occurs, try splitting the data to multiple smaller files.
    """

    def __init__(
        self,
        inference_task_name=BioNeMoConstants.TASK_INFERENCE,
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        super().__init__()

        self._inference_task_name = inference_task_name
        self.cfg = None
        self.app_root = None

    def _set_config(self, config):
        if not isinstance(config, dict):
            raise ValueError(f"Expected config to be of type dict but received type {type(config)}")

        # Received primitive dicts from server; convert back to OmegaConf
        if BioNeMoConstants.CONFIG in config:
            self.cfg = OmegaConf.create(config[BioNeMoConstants.CONFIG])
        else:
            raise ValueError(f"Received config did not contain BioNeMo config! Received keys: {list(config.keys())}")

    def _inference(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # First we extract DXO from the shareable.
        incoming_dxo = from_shareable(shareable)
        self._set_config(incoming_dxo.data)

        # Update loading paths with current app_root
        self.cfg.model.downstream_task.restore_from_path = os.path.join(
            self.app_root, self.cfg.model.downstream_task.restore_from_path
        )
        self.cfg.model.tokenizer.vocab_path = os.path.join(self.app_root, self.cfg.model.tokenizer.vocab_path)
        self.cfg.model.tokenizer.model_path = os.path.join(self.app_root, self.cfg.model.tokenizer.model_path)

        self.log_info(fl_ctx, "\n\n************** Experiment configuration ***********")
        self.log_info(fl_ctx, f"\n{OmegaConf.to_yaml(self.cfg)}")

        infer_class = import_class_by_path(self.cfg.infer_target)
        infer_model = infer_class(self.cfg)
        trainer = infer_model.trainer

        self.log_info(fl_ctx, "\n\n************** Restored model configuration ***********")
        self.log_info(fl_ctx, f"\n{OmegaConf.to_yaml(infer_model.model.cfg)}")

        # Update dataset_path to reflect client name
        client_name = fl_ctx.get_identity_name()
        self.cfg.model.data.dataset_path = os.path.join(self.cfg.model.data.dataset_path, f"data_{client_name}")

        # try to infer data_impl from the dataset_path file extension
        if self.cfg.model.data.dataset_path.endswith(".fasta"):
            self.cfg.model.data.data_impl = "fasta_fields_mmap"
        else:
            # Data are assumed to be CSV format if no extension provided
            self.log_info(fl_ctx, "File extension not supplied for data, inferring csv.")
            self.cfg.model.data.data_impl = "csv_fields_mmap"

        self.log_info(fl_ctx, f"Inferred data_impl: {self.cfg.model.data.data_impl}")

        if self.cfg.model.data.data_impl == "csv_fields_mmap":
            dataset_paths = expand_dataset_paths(self.cfg.model.data.dataset_path, ext=".csv")
            self.log_info(fl_ctx, f"Loading data from {dataset_paths}")
            ds = CSVFieldsMemmapDataset(
                dataset_paths, **self.cfg.model.data.data_impl_kwargs.get("csv_fields_mmap", {})
            )
        elif self.cfg.model.data.data_impl == "fasta_fields_mmap":
            dataset_paths = expand_dataset_paths(self.cfg.model.data.dataset_path, ext=".fasta")
            self.log_info(fl_ctx, f"Loading data from {dataset_paths}")
            ds = FASTAFieldsMemmapDataset(
                dataset_paths, **self.cfg.model.data.data_impl_kwargs.get("fasta_fields_mmap", {})
            )
        else:
            raise ValueError(f"Unknown data_impl: {self.cfg.model.data.data_impl}")

        # remove too long sequences
        filtered_ds = FilteredMappedDataset(
            dataset=ds,
            criterion_fn=lambda x: len(infer_model._tokenize([x["sequence"]])[0]) <= infer_model.model.cfg.seq_length,
        )

        dataloader = torch.utils.data.DataLoader(
            filtered_ds,
            batch_size=self.cfg.model.data.batch_size,
            num_workers=self.cfg.model.data.num_workers,
            drop_last=False,
        )

        # predict outputs for all sequences in batch mode
        all_batch_predictions = trainer.predict(
            model=infer_model,
            dataloaders=dataloader,
            return_predictions=True,
        )

        if not len(all_batch_predictions):
            raise ValueError("No predictions were made")

        # break batched predictions into individual predictions (list of dicts)
        predictions = []
        pred_keys = list(all_batch_predictions[0].keys())

        def cast_to_numpy(x):
            if torch.is_tensor(x):
                return x.cpu().numpy()
            return x

        for batch_predictions in all_batch_predictions:
            batch_size = len(batch_predictions[pred_keys[0]])
            for i in range(batch_size):
                predictions.append({k: cast_to_numpy(batch_predictions[k][i]) for k in pred_keys})

        # extract active hiddens if needed
        if "hiddens" in self.cfg.model.downstream_task.outputs:
            if ("hiddens" in predictions[0]) and ("mask" in predictions[0]):
                for p in predictions:
                    p["hiddens"] = p["hiddens"][p["mask"]]
                    del p["mask"]
        else:
            for p in predictions:
                del p["mask"]
                del p["hiddens"]

        # collect all results when using DDP
        self.log_info(fl_ctx, "Collecting results from all GPUs...")
        predictions = gather_objects(predictions, main_rank=0)
        # all but rank 0 will return None
        if predictions is None:
            return

        # from here only rank 0 should continue
        output_fname = self.cfg.model.data.output_fname
        if not output_fname:
            output_fname = f"{self.cfg.model.data.dataset_path}.pkl"
            self.log_info(fl_ctx, f"output_fname not specified, using {output_fname}")
        if os.path.exists(self.cfg.model.data.output_fname):
            self.log_warning(fl_ctx, f"Output path {output_fname} already exists, appending a unique id to the path")
            output_fname += "." + str(uuid.uuid4())

        n_sequences = len(predictions)
        self.log_info(fl_ctx, f"Saving {n_sequences} samples to output_fname = {output_fname}")
        pickle.dump(predictions, open(output_fname, "wb"))

        # Prepare a DXO for our updated model. Create shareable and return
        data_info = {BioNeMoConstants.NUMBER_SEQUENCES: n_sequences}

        outgoing_dxo = DXO(data_kind=BioNeMoConstants.DATA_INFO, data=data_info)
        return outgoing_dxo.to_shareable()

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self.log_info(fl_ctx, f"Task name: {task_name}")
        try:
            if task_name == self._inference_task_name:
                # get app root
                self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

                return self._inference(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
            else:
                # If unknown task name, set RC accordingly.
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in BioNeMoInferer execute: {secure_format_exception(e)}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
