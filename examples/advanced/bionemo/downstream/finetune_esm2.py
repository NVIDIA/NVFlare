# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Copied and adapted for NVFlare from https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py

import argparse
import random
from pathlib import Path
from lightning import seed_everything
from typing import Dict, List, Optional, Sequence, Tuple, Type, get_args

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import (
    InMemoryPerTokenValueDataset,
    InMemoryProteinDataset,
    InMemorySingleValueDataset,
)
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger

# Resue parser and config constants from bionemo
from bionemo.esm2.scripts.finetune_esm2 import get_parser, SUPPORTED_CONFIGS, SUPPORTED_DATASETS

# (1) import nvflare lightning client API
import nvflare.client.lightning as flare


def train_model(
    train_data_path: Path,
    valid_data_path: Path,
    num_nodes: int,
    devices: int,
    min_seq_length: Optional[int],
    max_seq_length: int,
    result_dir: Path,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    log_every_n_steps: Optional[int],
    num_dataset_workers: int,
    lr: float,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    task_type: str = "regression",
    encoder_frozen: bool = False,
    scale_lr_layer: Optional[str] = None,
    lr_multiplier: float = 1.0,
    # single value classification / regression mlp
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    # token-level classification cnn
    cnn_dropout: float = 0.25,
    cnn_hidden_size: int = 32,
    cnn_num_classes: int = 3,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: Optional[bool] = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    restore_from_checkpoint_path: Optional[str] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    dataset_class: Type[InMemoryProteinDataset] = InMemorySingleValueDataset,
    config_class: Type[BioBertConfig] = ESM2FineTuneSeqConfig,
    metric_tracker: Callback | None = None,
    overlap_grad_reduce: bool = False,  # Default to False to avoid communication issue in gradient synchronization step
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    grad_reduce_in_fp32: bool = False,
) -> Tuple[Path, Callback | None, nl.Trainer]:
    """Train an ESM2 model on UR data.

    Args:
        train_data_path (Path): path to train CSV
        valid_data_path (Path): path to validation CSV
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        min_seq_length (Optional[int]): minimum sequence length
        max_seq_length (int): maximum sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss
        log_every_n_steps (Optional[int]): log every n steps
        num_dataset_workers (int): number of dataset workers
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        accumulate_grad_batches (int): number of batches to accumulate gradients for
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        precision (PrecisionTypes): Precision type for training (e.g., float16, float32)
        task_type (Literal["classification", "regression"]): Fine-tuning task type. Default is regression.
        encoder_frozen (bool): Freeze the encoder parameters. Default is False.
        scale_lr_layer (Optional[str]): layer names for which the lr is scaled by lr_multiplier
        lr_multiplier (float): lr multiplier for parameters in scale_lr_layer
        mlp_ft_dropout (float): dropout for single value classification / regression mlp
        mlp_hidden_size (int): dimension of hidden layer in mlp task head
        mlp_target_size: (int): output dimension of the mlp task head (number of classes in classification tasks)
        cnn_dropout (float): dropout for token-level classification cnn
        cnn_hidden_size (int): hidden dimension of cnn head
        cnn_num_classes (int): number of classes in token-level classification
        wandb_entity (Optional[str]): The team posting this run (default: your username or your default team)
        wandb_project (Optional[str]): The name of the project to which this run will belong
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_tags (Optional[List[str]]): Tags associated with this run
        wandb_group (Optional[str]): A unique string shared by all runs in a given group
        wandb_id (Optional[str]): Sets the version, mainly used to resume a previous run
        wandb_anonymous (Optional[bool]): Enables or explicitly disables anonymous logging
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers
        pipeline_model_parallel_size (int): pipeline model parallel size
        tensor_model_parallel_size (int): tensor model parallel size
        create_tensorboard_logger (bool): create the tensorboard logger
        restore_from_checkpoint_path (Optional[str]): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
        save_last_checkpoint (bool): whether to save the last checkpoint
        metric_to_monitor_for_checkpoints (str): metric to monitor for checkpoints
        save_top_k (int): number of top checkpoints to save
        nsys_profiling (bool): whether to enable nsys profiling
        nsys_start_step (int): start step for nsys profiling
        nsys_end_step (Optional[int]): end step for nsys profiling
        nsys_ranks (List[int]): ranks for nsys profiling
        dataset_class (Type[InMemoryProteinDataset]): The dataset class for loading the data from a CSV file
        config_class (Type[BioBertConfig]): The config class for configuring the model using checkpoint provided
        metric_tracker: Optional callback to track metrics (used for testing)
        overlap_grad_reduce (bool): overlap gradient reduction
        overlap_param_gather (bool): overlap parameter gather
        average_in_collective (bool): average in collective
        grad_reduce_in_fp32 (bool): gradient reduction in fp32
    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=False,  # do not use `ckpt_async_save=True` as the checkpoint might still be saved while the next round already removed that saving directory
        ckpt_parallel_load=True,
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/lightning.pytorch.loggers.html"
    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
    ]
    if metric_tracker is not None:
        callbacks.append(metric_tracker)
    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )

    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )

    tokenizer = get_tokenizer()

    # Initialize the data module.
    train_dataset = dataset_class.from_csv(train_data_path, task_type=task_type)
    valid_dataset = dataset_class.from_csv(valid_data_path, task_type=task_type)
    if task_type == "classification":
        valid_dataset.label_tokenizer = train_dataset.label_tokenizer
        
    data_module = ESM2FineTuneDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        tokenizer=tokenizer,
    )
    # Configure the model
    train_metric = None
    if task_type == "regression":
        valid_metric = TorchmetricsConfig(class_path="MeanSquaredError", task="regression", metric_name="val_mse")
    else:
        valid_metric = TorchmetricsConfig(
            class_path="Accuracy",
            task="classification",
            kwargs={
                "task": "multiclass",
                "threshold": 0.5,
                "num_classes": data_module.train_dataset.label_tokenizer.vocab_size,
            },
            metric_name="val_acc",
        )

    if tensor_model_parallel_size * pipeline_model_parallel_size > 1 and (
        train_metric is not None or valid_metric is not None
    ):
        raise NotImplementedError("Metric logging under model parallelism is not supported yet.")

    config = config_class(
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
        train_metric=train_metric,
        valid_metric=valid_metric,
    )
    # Mapping of task-dependent config attributes to their new values
    task_dependent_attr = {
        "mlp_ft_dropout": mlp_ft_dropout,
        "mlp_hidden_size": mlp_hidden_size,
        "mlp_target_size": mlp_target_size,
        "cnn_dropout": cnn_dropout,
        "cnn_hidden_size": cnn_hidden_size,
        "cnn_num_classes": cnn_num_classes,
    }
    # Update attributes only if they exist in the config
    for attr, value in task_dependent_attr.items():
        if hasattr(config, attr):
            setattr(config, attr, value)

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",  # fused_adam not supported
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
        ),
    )
    # fiddle is not serializing lambda fn
    # to bypass serialization of lambda fn scale_lr_condition as part of optimizer configuration
    if scale_lr_layer:
        optimizer.scale_lr_cond = lambda name, param: scale_lr_layer in name
        optimizer.lr_mult = lr_multiplier

    module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)

    #If client should save best local checkpoints, set to `save_local_ckpt=True`,  
    save_local_ckpt = False
    if save_local_ckpt:
        # Configure our custom Checkpointer
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
            filename="checkpoint-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
        )
    else:
        checkpoint_callback = None

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )

    # (2) patch the lightning trainer
    flare.patch(trainer, restore_state=False, load_state_dict_strict=False, update_fit_loop=False)

    while flare.is_running():    
        # (3) receives FLModel from NVFlare
        # Note that we don't need to pass this input_model to trainer
        # because after flare.patch the trainer.fit/validate will get the
        # global model internally

        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}, Global model = {input_model} ({len(input_model.params)} params)]\n")

        # (4) evaluate the current global model to allow server-side model selection
        #print("--- validate global model ---")
        #trainer.validate(module, datamodule=data_module)

        # perform local training starting with the received global model
        print("--- train new model ---")
        if input_model.current_round == 0:
            # Use llm.train only in first round as it includes additional setup
            llm.train(
                model=module,
                data=data_module,
                trainer=trainer,
                log=nemo_logger,
                resume=None,
            )
        else:
            # make sure the data loaders are reshuffling the data each round
            seed = random.randint(0, 10000)
            seed_everything(seed) 
            print("Resetting seed {seed}")
            
            trainer.fit(module, datamodule=data_module)
            
    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, metric_tracker, trainer


def finetune_esm2_entrypoint():
    """Entrypoint for running ESM2 finetuning."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()

    # to avoid padding for single value labels:
    if args.min_seq_length is not None and args.datset_class is InMemorySingleValueDataset:
        parser.error("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")

    # 2. Call pretrain with args
    train_model(
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        result_dir=args.result_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        num_dataset_workers=args.num_dataset_workers,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        task_type=args.task_type,
        encoder_frozen=args.encoder_frozen,
        scale_lr_layer=args.scale_lr_layer,
        lr_multiplier=args.lr_multiplier,
        # single value classification / regression mlp
        mlp_ft_dropout=args.mlp_ft_dropout,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_target_size=args.mlp_target_size,
        # token-level classification cnn
        cnn_dropout=args.cnn_dropout,
        cnn_hidden_size=args.cnn_hidden_size,
        cnn_num_classes=args.cnn_num_classes,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        restore_from_checkpoint_path=args.restore_from_checkpoint_path,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        nsys_profiling=args.nsys_profiling,
        nsys_start_step=args.nsys_start_step,
        nsys_end_step=args.nsys_end_step,
        nsys_ranks=args.nsys_ranks,
        dataset_class=args.dataset_class,
        config_class=args.config_class,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=not args.no_overlap_param_gather,
        average_in_collective=not args.no_average_in_collective,
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
    )


if __name__ == "__main__":
    finetune_esm2_entrypoint()
