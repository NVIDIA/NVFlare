import argparse

import datasets
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
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
        default="llama2-7b-dolly-sft",
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

    # Training arguments
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=3,
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

    # Evaluate
    trainer.evaluate()

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
