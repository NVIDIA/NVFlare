import argparse
import os

import datasets
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    utils,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, trainer_utils
from trl import SFTTrainer

use_flash_attention = True


def format_instruction(example):
    output_texts = []
    for i in range(len(example["input"])):
        text = f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input'][i]} ### Response: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/model/llama-2-7b-hf",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="/dataset/dolly/training.jsonl",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="/dataset/dolly/validation.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llama2-7b-dolly-sft-iter",
    )
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()

    # Dataset
    dataset_train = datasets.load_dataset("json", data_files=args.data_path_train, split="train")
    dataset_valid = datasets.load_dataset("json", data_files=args.data_path_valid, split="train")
    # Print dataset info
    print(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")
    # record every 5% of the dataset
    batch_size = 4
    gra_accu_steps = 10
    logging_steps = int(len(dataset_train) / (20 * batch_size * gra_accu_steps))
    print(f"logging_steps: {logging_steps}")

    # Model configs
    model_path = args.model_path
    if args.mode:
        # If PEFT, set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            use_cache=False,
            use_flash_attention_2=use_flash_attention,
            device_map="auto",
        )
        # prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_flash_attention_2=use_flash_attention,
            use_cache=False,
            device_map="auto",
        )

    model.config.pretraining_tp = 1
    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Save base model state_dict, which will be used as the starting
    # weights for each round - to show the weights are loaded correctly
    if args.mode:
        params = get_peft_model_state_dict(model)
    else:
        params = model.state_dict()
    torch.save(params, "model_dict_base.pt")

    # Training arguments
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True,
    )

    # Trainer
    max_seq_length = 2048
    if args.mode:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=False,
            formatting_func=format_instruction,
            args=train_args,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=False,
            formatting_func=format_instruction,
            args=train_args,
        )

    # Train iteratively by using "resume" functionality
    # and replace the resume weights every round
    for curr_round in range(3):
        print(f"current_round={curr_round}")

        # Evaluate
        state_dict_replace = torch.load("model_dict_base.pt", map_location="cpu")
        if args.mode:
            set_peft_model_state_dict(trainer.model, state_dict_replace)
        else:
            trainer.model.load_state_dict(state_dict_replace)
        trainer.evaluate()

        # Train
        if curr_round == 0:
            # First round, start from pretrained model
            trainer.train()
        else:
            # replace local resume weights with global weights
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            if args.mode:
                # PEFT model small, directly save via torch.save
                resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                torch.save(state_dict_replace, resume_model_file_path)
            else:
                # SFT model can be large, save via HF API
                trainer.model.save_pretrained(resume_from_checkpoint_folder)
            # increment num_train_epochs so that the trainer will continue training
            trainer.args.num_train_epochs += 1
            # continue training
            trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()
