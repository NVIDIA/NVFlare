import argparse
import copy
import os

import datasets
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict, utils
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, trainer_utils
from trl import SFTTrainer

import nvflare.client as flare

use_flash_attention = False


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
        default="./model/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="./dataset/dolly/training.jsonl",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="./dataset/dolly/validation.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./workspace_fl/llama2-7b-dolly-sft",
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
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
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

    # initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # receives FLModel from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"current_round={curr_round}")

        # fix the key name received from global model if using model def file
        global_model = copy.deepcopy(input_model.params)
        for key in list(global_model.keys()):
            global_model[key.replace("model.", "", 1)] = global_model.pop(key)

        # wraps evaluation logic into a method to re-use for
        # evaluation on both trained and received model
        def evaluate(input_weights, mode):
            if mode:
                set_peft_model_state_dict(trainer.model, input_weights)
            else:
                trainer.model.load_state_dict(input_weights)
            metric_score = trainer.evaluate()
            print(f"Evaluation metric score: {metric_score}")
            return metric_score

        # evaluate on received global model
        eval_loss = evaluate(global_model, args.mode)
        eval_loss = float(eval_loss["eval_loss"])

        # loads global model
        # Since we perform iterative training by using "resume" functionality
        # we need to replace the resume weights with global weights every round
        if curr_round == 0:
            # First round, start from pretrained model
            trainer.train()
        else:
            # replace local resume weights with global weights
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            if args.mode:
                # PEFT model small, directly save via torch.save
                resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                torch.save(global_model, resume_model_file_path)
            else:
                # SFT model can be large, save via HF API
                trainer.model.save_pretrained(resume_from_checkpoint_folder)
            # increment num_train_epochs so that the trainer will continue training
            trainer.args.num_train_epochs += 1
            # continue training
            trainer.train(resume_from_checkpoint=True)

        # compose output model to send back to server
        checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
        if args.mode:
            # PEFT, load PEFT part from current checkpointing folder
            model_file_path = os.path.join(checkpoint_folder, utils.WEIGHTS_NAME)
            out_param = torch.load(model_file_path, map_location="cpu")
        else:
            # SFT, load whole model state_dict
            out_param = trainer.model.state_dict()

        # fix the key name sent to global model
        if not args.mode:
            for key in list(out_param.keys()):
                out_param["model." + key] = out_param.pop(key).cpu()

        # construct trained FL model
        output_model = flare.FLModel(
            params=out_param,
            metrics={"eval_loss": eval_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": trainer.train_dataset.num_rows},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
