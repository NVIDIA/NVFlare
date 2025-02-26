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

import argparse
import os

import pandas as pd
import torch
from data_sequence import DataSequence
from nlp_models import BertModel, GPTModel
from seqeval.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import nvflare client API
import nvflare.client as flare

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, nargs="?")
    parser.add_argument("--batch_size", type=int, default=16, nargs="?")
    parser.add_argument("--learning_rate", type=float, default=1e-5, nargs="?")
    parser.add_argument("--num_workers", type=int, default=1, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=1, nargs="?")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", nargs="?")
    parser.add_argument("--num_labels", type=int, default=3, nargs="?")
    parser.add_argument("--ignore_token", type=int, default=-100, nargs="?")
    return parser.parse_args()


def get_labels(df_train, num_labels):
    labels = []
    for x in df_train["labels"].values:
        labels.extend(x.split(" "))
    unique_labels = set(labels)
    # check label length
    if len(unique_labels) != num_labels:
        raise ValueError(f"num_labels {num_labels} need to align with dataset, actual data {len(unique_labels)}!")
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return labels_to_ids, ids_to_labels


def main():
    # define local parameters
    args = define_parser()
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.learning_rate
    num_workers = args.num_workers
    local_epochs = args.local_epochs
    model_name = args.model_name
    num_labels = args.num_labels
    ignore_token = args.ignore_token

    # Initializes NVFlare client API and get site_name from flare
    flare.init()
    site_name = flare.get_site_name()

    # load data
    df_train = pd.read_csv(os.path.join(dataset_path, site_name + "_train.csv"))
    df_valid = pd.read_csv(os.path.join(dataset_path, site_name + "_val.csv"))
    labels_to_ids, ids_to_labels = get_labels(df_train, num_labels)

    # training components
    writer = SummaryWriter("./")
    if model_name == "bert-base-uncased":
        model = BertModel(model_name=model_name, num_labels=num_labels)
    elif model_name == "gpt2":
        model = GPTModel(model_name=model_name, num_labels=num_labels)
    else:
        raise ValueError(f"Model {model_name} not supported!")
    tokenizer = model.tokenizer
    train_dataset = DataSequence(df_train, labels_to_ids, tokenizer=tokenizer, ignore_token=ignore_token)
    valid_dataset = DataSequence(df_valid, labels_to_ids, tokenizer=tokenizer, ignore_token=ignore_token)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    print(f"Training Size: {len(train_loader.dataset)}, Validation Size: {len(valid_loader.dataset)}")
    optimizer = AdamW(model.parameters(), lr=lr)
    local_model_file = "local_model.pt"
    best_global_model_file = "best_global_model_file.pt"
    best_acc = 0.0

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receive FLModel from NVFlare
        global_model = flare.receive()
        curr_round = global_model.current_round
        epoch_global = local_epochs * curr_round
        print(f"({site_name}) current_round={curr_round + 1}/{global_model.total_rounds}")

        # load global model from NVFlare
        model.load_state_dict(global_model.params)
        model.to(DEVICE)

        # wraps evaluation logic into a method to re-use for
        # evaluation on both trained and received model
        def evaluate(tb_id):
            model.eval()
            with torch.no_grad():
                total_acc_val, total_loss_val, val_total = 0, 0, 0
                val_y_pred, val_y_true = [], []
                for val_data, val_label in valid_loader:
                    val_label = val_label.to(DEVICE)
                    val_total += val_label.shape[0]
                    mask = val_data["attention_mask"].squeeze(1).to(DEVICE)
                    input_id = val_data["input_ids"].squeeze(1).to(DEVICE)
                    # Inference
                    loss, logits = model(input_id, mask, val_label)
                    # Add items for metric computation
                    for i in range(logits.shape[0]):
                        # remove pad tokens
                        logits_clean = logits[i][val_label[i] != ignore_token]
                        label_clean = val_label[i][val_label[i] != ignore_token]
                        # calcluate acc and store prediciton and true labels
                        predictions = logits_clean.argmax(dim=1)
                        acc = (predictions == label_clean).float().mean()
                        total_acc_val += acc.item()
                        val_y_pred.append([ids_to_labels[x.item()] for x in predictions])
                        val_y_true.append([ids_to_labels[x.item()] for x in label_clean])
                # compute metric
                metric_dict = classification_report(
                    y_true=val_y_true, y_pred=val_y_pred, output_dict=True, zero_division=0
                )
                # tensorboard record id prefix, add to record if provided
                writer.add_scalar(tb_id + "_precision", metric_dict["macro avg"]["precision"], epoch_global)
                writer.add_scalar(tb_id + "_recall", metric_dict["macro avg"]["recall"], epoch_global)
                writer.add_scalar(tb_id + "_f1-score", metric_dict["macro avg"]["f1-score"], epoch_global)
            return metric_dict["macro avg"]["f1-score"]

        # evaluate on received global model
        val_acc = evaluate("global_val_acc")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_global_model_file)

        # train local model
        epoch_len = len(train_loader)
        for epoch in range(local_epochs):
            model.train()
            print(f"Local epoch {site_name}: {epoch + 1}/{local_epochs} (lr={lr})")

            for i, batch_data in enumerate(train_loader):
                mask = batch_data[0]["attention_mask"].squeeze(1).to(DEVICE)
                input_id = batch_data[0]["input_ids"].squeeze(1).to(DEVICE)
                train_label = batch_data[1].to(DEVICE)
                # model output
                loss, logits = model(input_id, mask, train_label)
                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record loss
                current_step = epoch_len * epoch_global + i
                writer.add_scalar("train_loss", loss.item(), current_step)

        # evaluation on local trained model
        val_acc_local = evaluate("local_val_acc")
        torch.save(model.state_dict(), local_model_file)

        # construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"eval_acc": val_acc_local},
            meta={"NUM_STEPS_CURRENT_ROUND": epoch_len * local_epochs},
        )

        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
