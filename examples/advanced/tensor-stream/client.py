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
from model import get_model
from trainer import get_dataset, get_training_arguments, preprocess

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def main():
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # Load pretrained model + tokenizer
    model, tokenizer = get_model("gpt2")

    # Load dataset and preprocess
    dataset = get_dataset()
    train_dataset = dataset["train"].map(lambda x: preprocess(x, tokenizer), batched=True)

    training_args = get_training_arguments(client_name)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    summary_writer = SummaryWriter()

    while flare.is_running():
        input_model = flare.receive()
        print(f"site={client_name}, current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(training_args.device)

        # comment out the next line to skip training, and only do model exchange
        trainer.train()

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"ROUND": input_model.current_round},
        )
        print(f"site={client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    main()
