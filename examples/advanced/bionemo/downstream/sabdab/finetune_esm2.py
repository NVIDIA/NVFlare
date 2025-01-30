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
from pathlib import Path
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
from bionemo.esm2.model.finetune.finetune_regressor import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.finetune_token_classifier import ESM2FineTuneTokenConfig
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


__all__: Sequence[str] = ("train_model", "finetune_esm2_entrypoint", "get_parser")


SUPPORTED_CONFIGS = {
    "ESM2FineTuneSeqConfig": ESM2FineTuneSeqConfig,
    "ESM2FineTuneTokenConfig": ESM2FineTuneTokenConfig,
}

SUPPORTED_DATASETS = {
    "InMemoryProteinDataset": InMemoryProteinDataset,
    "InMemorySingleValueDataset": InMemorySingleValueDataset,
    "InMemoryPerTokenValueDataset": InMemoryPerTokenValueDataset,
}

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
    encoder_frozen: bool = False,
    scale_lr_layer: Optional[str] = None,
    lr_multiplier: float = 1.0,
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
        encoder_frozen (bool): Freeze the encoder parameters. Default is False.
        scale_lr_layer (Optional[str]): layer names for which the lr is scaled by lr_multiplier
        lr_multiplier (float): lr multiplier for parameters in scale_lr_layer
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
    print("XXXXXX starting train_model")
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

    print("XXXXXX starting MegatronStrategy")
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
        ckpt_async_save=True,
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

    print("XXXXXX starting Trainer")

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
    # (2) patch the lightning trainer
    flare.patch(trainer, restore_state=False, load_state_dict_strict=False)

    tokenizer = get_tokenizer()

    # Initialize the data module.
    train_dataset = dataset_class.from_csv(train_data_path)
    valid_dataset = dataset_class.from_csv(valid_data_path)

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
    config = config_class(
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        # initial_ckpt_skip_keys_with_these_prefixes=[],  # load everything from the checkpoint.
    )

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

    # Configure our custom Checkpointer
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=save_last_checkpoint,
        monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
        save_top_k=save_top_k,
        every_n_train_steps=val_check_interval,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        filename="checkpoint-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )

    # (3) receives FLModel from NVFlare
    # Note that we don't need to pass this input_model to trainer
    # because after flare.patch the trainer.fit/validate will get the
    # global model internally
    input_model = flare.receive()
    print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")
    
    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=None, #nemo_logger,
        resume=None #resume.AutoResume(
            #resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            #resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        #),
    )

    flare.shutdown()

    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
        
    return ckpt_path, metric_tracker, trainer


def finetune_esm2_entrypoint():
    """Entrypoint for running ESM2 finetuning."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()
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
        encoder_frozen=args.encoder_frozen,
        scale_lr_layer=args.scale_lr_layer,
        lr_multiplier=args.lr_multiplier,
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


def get_parser():
    """Return the cli parser for this tool."""
    # TODO migrate to hydra config
    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    parser = argparse.ArgumentParser(description="Pretrain ESM2 with UR data.")
    parser.add_argument(
        "--train-data-path",
        type=Path,
        required=True,
        help="Path to the train data CSV file",
    )
    parser.add_argument(
        "--valid-data-path",
        type=Path,
        required=True,
        help="Path to the valid data CSV file",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--encoder-frozen",
        action="store_true",
        default=False,
        help="Freeze the encoder parameters",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=4e-4,
        help="Learning rate for training. Default is 4e-4",
    )
    parser.add_argument(
        "--scale-lr-layer",
        type=str,
        required=False,
        default=None,
        help="Layer name for which we scale the lr by lr-multiplier",
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        required=False,
        default=1.0,
        help="Learning rate multiplier for layers with scale-lr-layer in their name",
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    # FIXME (@skothenhill) figure out how checkpointing and resumption should work with the new nemo trainer
    parser.add_argument(
        "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument("--experiment-name", type=str, required=False, default="esm2", help="Name of the experiment.")

    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=500000,
        help="Number of steps to use for training. Default is 500000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to use for training. Default is 1.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps between validation. Default is 10000.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        required=False,
        help="Number of steps between logging. Default is 50.",
    )
    parser.add_argument(
        "--min-seq-length",
        type=float_or_int_or_none,
        required=False,
        default=1024,
        help="Minimum sequence length. Sampled will be padded if less than this value. Set 'None' to unset minimum.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=False,
        default=1024,
        help="Maximum sequence length. Samples will be truncated if exceeds this value.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float_or_int_or_none,
        required=False,
        default=2,
        help="Number of global batches used for validation if int. Fraction of validation dataset if float. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        required=False,
        default=1,
        help="Gradient accumulation steps. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action="store_true",
        default=True,
        help="Save the last checkpoint.",
    )
    parser.add_argument(
        "--metric-to-monitor-for-checkpoints",
        type=str,
        required=False,
        default="val_loss",
        help="The metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        required=False,
        default=2,
        help="Save the top k checkpoints.",
    )
    parser.add_argument(
        "--restore-from-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the checkpoint directory to restore from. Will override `--resume-if-exists` when set.",
    )
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )
    # DDP config
    parser.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-overlap-param-gather",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-average-in-collective",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--grad-reduce-in-fp32",
        action="store_true",
        default=False,
    )

    config_class_options: Dict[str, Type[BioBertConfig]] = SUPPORTED_CONFIGS

    def config_class_type(desc: str) -> Type[BioBertConfig]:
        try:
            return config_class_options[desc]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Do not recognize key {desc}, valid options are: {config_class_options.keys()}"
            )

    parser.add_argument(
        "--config-class",
        type=config_class_type,
        default=ESM2FineTuneSeqConfig,
        help="Model configs link model classes with losses, and handle model initialization (including from a prior "
        "checkpoint). This is how you can fine-tune a model. First train with one config class that points to one model "
        "class and loss, then implement and provide an alternative config class that points to a variant of that model "
        "and alternative loss. In the future this script should also provide similar support for picking different data "
        f"modules for fine-tuning with different data types. Choices: {config_class_options.keys()}",
    )

    dataset_class_options: Dict[str, Type[InMemoryProteinDataset]] = SUPPORTED_DATASETS

    def dataset_class_type(desc: str) -> Type[InMemoryProteinDataset]:
        try:
            return dataset_class_options[desc]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Do not recognize key {desc}, valid options are: {dataset_class_options.keys()}"
            )

    parser.add_argument(
        "--dataset-class",
        type=dataset_class_type,
        default=InMemorySingleValueDataset,
        help=f"Dataset class name for finetuning. Choices: {config_class_options.keys()}",
    )
    return parser


if __name__ == "__main__":
    finetune_esm2_entrypoint()
