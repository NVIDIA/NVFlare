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
import logging
import random
import time

import datasets
import torch
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

from nvflare.apis.dxo import DXO, DataKind, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Add callback to stop at each epoch
class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


def format_instruction(example):
    return f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input']} ### Response: {example['output']}"


class HFSFTTaskProcessor(DeviceTaskProcessor):

    def __init__(
        self,
        model_name_or_path: str,
        data_path_train: str,
        data_path_valid: str,
        output_path: str,
        communication_delay: dict,
        device_speed: dict,
        subset_size: int = None,
        local_epochs: int = 1,
        local_batch_size: int = 4,
        local_lr: float = 5e-4,
        lr_scheduler: str = "constant",
    ):
        DeviceTaskProcessor.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.data_path_train = data_path_train
        self.data_path_valid = data_path_valid
        self.output_path = output_path
        self.communication_delay = communication_delay
        self.device_speed = device_speed
        self.subset_size = subset_size
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.local_lr = local_lr
        self.lr_scheduler = lr_scheduler

        # Device speed type for simulation
        mean_speed = self.device_speed.get("mean")
        self.device_speed_type = random.randint(0, len(mean_speed) - 1)

        # Training
        self.model = None
        self.trainer = None
        self.dataset_train = None
        self.dataset_valid = None

    def _create_data_subset(self, dataset, subset_size):
        """Create a data subset for the specific device"""
        if self.subset_size is None:
            # If no subset size specified, use the full dataset
            return dataset

        dataset_size = len(dataset)
        if subset_size >= dataset_size:
            # If subset size is larger than dataset, use full dataset
            self.logger.warning(f"Subset size {subset_size} >= dataset size {dataset_size}, using full dataset")
            return dataset

        # Create indices list and shuffle with device-specific seed
        indices = list(range(dataset_size))
        random.shuffle(indices)
        # Select the first subset_size indices
        subset_indices = indices[:subset_size]
        # Create subset dataset
        subset_data = dataset.select(subset_indices)
        return subset_data

    def setup(self, job: JobResponse) -> None:
        device_id = self.device.device_id if self.device else "unknown"
        self.logger.info(f"Device {device_id}: Starting setup...")

        # Dataset
        self.logger.info(f"Device {device_id}: Loading datasets...")
        full_dataset_train = datasets.load_dataset("json", data_files=self.data_path_train, split="train")
        full_dataset_valid = datasets.load_dataset("json", data_files=self.data_path_valid, split="train")
        self.logger.info(
            f"Device {device_id}: Loaded full datasets - train: {len(full_dataset_train)}, valid: {len(full_dataset_valid)}"
        )
        # Subset train dataset if subset_size is specified
        self.dataset_train = self._create_data_subset(full_dataset_train, self.subset_size)
        valid_train_ratio = len(full_dataset_valid) / len(full_dataset_train)
        self.dataset_valid = self._create_data_subset(full_dataset_valid, int(self.subset_size * valid_train_ratio))

        # Load model
        self.logger.info(f"Device {device_id}: Loading model {self.model_name_or_path}...")
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
        torch.set_default_dtype(default_dtype)
        self.model.config.pretraining_tp = 1
        self.logger.info(
            f"Device {device_id}: Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

        # Training arguments
        train_args = SFTConfig(
            output_dir=self.output_path + f"/device_{device_id}",
            num_train_epochs=self.local_epochs * 3,
            per_device_train_batch_size=self.local_batch_size,
            gradient_accumulation_steps=10,
            gradient_checkpointing=False,
            optim="adamw_torch",
            logging_steps=logging_steps,
            save_strategy="epoch",
            learning_rate=self.local_lr,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type=self.lr_scheduler,
            disable_tqdm=True,
            max_length=1024,
            save_total_limit=2,
            save_safetensors=False,
            seed=0,
            data_seed=0,
            # Add settings to help with shutdown
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,  # Prevent dataset modification issues
        )

        # Create trainer
        self.logger.info(f"Device {device_id}: Creating SFT trainer...")
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_valid,
            peft_config=None,  # No PEFT for SFT mode
            formatting_func=format_instruction,
            args=train_args,
            callbacks=[StopCallback()],
        )
        self.logger.info(f"Device {device_id}: Setup completed successfully!")

    def shutdown(self) -> None:
        """Clean up resources"""
        # Clean up trainer first
        if self.trainer:
            # Try to properly cleanup trainer components
            if hasattr(self.trainer, "model") and self.trainer.model:
                self.trainer.model.cpu()  # Move model to CPU
            if hasattr(self.trainer, "tokenizer"):
                del self.trainer.tokenizer
            del self.trainer
            self.trainer = None

        # Clean up model
        if self.model:
            self.model.cpu()  # Move to CPU to free GPU memory
            del self.model
            self.model = None

        # Clean up datasets
        if hasattr(self, "dataset_train") and self.dataset_train:
            del self.dataset_train
            self.dataset_train = None
        if hasattr(self, "dataset_valid") and self.dataset_valid:
            del self.dataset_valid
            self.dataset_valid = None

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sft_training(self, global_model):
        """Perform SFT training with the global model"""
        device_id = self.device.device_id if self.device else "unknown"

        # Load the global model state dict
        self.logger.info(f"Device {device_id}: Loading global model weights...")
        self.trainer.model.load_state_dict(global_model)

        # Evaluate the global model
        self.logger.info(f"Device {device_id}: Evaluating global model...")
        eval_result = self.trainer.evaluate()
        eval_loss = float(eval_result["eval_loss"])
        self.logger.info(f"Device {device_id}: Global model eval_loss: {eval_loss:.4f}")

        # Train for specified number of local epochs
        self.logger.info(f"Device {device_id}: Starting local training for {self.local_epochs} epochs...")
        self.logger.info(f"Device {device_id}: Training dataset size: {len(self.dataset_train)} samples")
        # Use resume from to do multi-epoch training
        resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
        trainer.model.save_pretrained(resume_from_checkpoint_folder, state_dict=global_model, safe_serialization=False)

        for epoch in range(self.local_epochs):
            self.trainer.train(resume_from_checkpoint=True)

        self.logger.info(f"Device {device_id}: Local training completed")

        # Get the trained model state dict
        trained_model = self.trainer.model.state_dict()

        # Return the full trained model state dict instead of calculating differences
        # The differences will be calculated on the server side for better performance
        result_dict = {}
        for key, param in trained_model.items():
            # Convert to CPU and numpy for serialization
            if hasattr(param, "cpu"):
                param_cpu = param.cpu()
                # Keep as numpy array (not list) for faster serialization
                result_dict[key] = param_cpu.numpy()
            else:
                result_dict[key] = param

        self.logger.info(f"Device {device_id}: Prepared {len(result_dict)} parameters for transmission")
        return result_dict

    def process_task(self, task: TaskResponse) -> dict:
        device_id = self.device.device_id if self.device else "unknown"
        task_start_time = time.time()
        self.logger.info(f"Device {device_id}: ========== Task received at {time.strftime('%H:%M:%S')} ==========")

        # Deserialize task data
        deserialize_start = time.time()
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        deserialize_time = time.time() - deserialize_start
        self.logger.info(f"Device {device_id}: Task deserialization took {deserialize_time:.2f} seconds")

        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        if model.data_kind != DataKind.WEIGHTS:
            self.logger.error(f"expect model data kind to be {DataKind.WEIGHTS} but got {model.data_kind}")
            raise ValueError("bad model data kind")

        global_model = model.data
        if not isinstance(global_model, dict):
            self.logger.error(f"expect global_model to be dict but got {type(global_model)}")
            raise ValueError("bad global model")

        self.logger.info(f"Device {device_id}: Received global model with {len(global_model)} parameter groups")

        # Convert the global_model to a dict of tensors
        conversion_start = time.time()
        self.logger.info(f"Device {device_id}: Converting global model to tensors...")
        global_model = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in global_model.items()}
        conversion_time = time.time() - conversion_start
        self.logger.info(f"Device {device_id}: Tensor conversion took {conversion_time:.2f} seconds")

        # Perform SFT training
        self.logger.info(f"Device {device_id}: Starting SFT training...")
        trained_weights = self._sft_training(global_model)

        # Create the result DXO with trained weights
        self.logger.info(f"Device {device_id}: Creating result DXO with {len(trained_weights)} parameters")
        result_dxo = DXO(data_kind=DataKind.WEIGHTS, data=trained_weights)  # Send full weights, not diffs

        # Simulate training and communication delays
        self.logger.info(f"Device {device_id}: Calculating simulation delays...")
        # Random delay to simulate the training time difference
        mean_speed = self.device_speed.get("mean")
        std_speed = self.device_speed.get("std")

        if len(mean_speed) != len(std_speed):
            self.logger.error("mean and std speed should be of same length")
            raise ValueError("bad device speed data")

        # Sample the training delay from the normal distribution
        delay_speed = random.gauss(mean_speed[self.device_speed_type], std_speed[self.device_speed_type])
        if delay_speed < 0:
            delay_speed = 0

        # Get communication delay
        mean_comm = self.communication_delay.get("mean")
        std_comm = self.communication_delay.get("std")
        delay_comm = random.gauss(mean_comm, std_comm)
        if delay_comm < 0:
            delay_comm = 0

        total_delay = delay_speed + delay_comm

        # Log the delay with the device id
        self.logger.info(
            f"Device {device_id}: Simulating additional delay: {total_delay:.2f} seconds (training: {delay_speed:.2f}s, comm: {delay_comm:.2f}s)"
        )

        time.sleep(total_delay)

        # Serialize result
        serialize_start = time.time()
        result_dict = result_dxo.to_dict()
        serialize_time = time.time() - serialize_start
        self.logger.info(f"Device {device_id}: Result serialization took {serialize_time:.2f} seconds")

        total_task_time = time.time() - task_start_time
        self.logger.info(
            f"Device {device_id}: ========== Task completed in {total_task_time:.2f} seconds at {time.strftime('%H:%M:%S')} =========="
        )
        return result_dict
