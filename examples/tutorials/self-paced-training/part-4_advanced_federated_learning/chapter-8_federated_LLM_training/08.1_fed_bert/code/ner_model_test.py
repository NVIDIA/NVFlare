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

import argparse
import os

import pandas as pd
import torch
from seqeval.metrics import classification_report
from src.data_sequence import DataSequence
from src.nlp_models import BertModel, GPTModel
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Perform model testing by loading the best global model")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--model_path", type=str, help="Path to workspace server folder")
    parser.add_argument("--num_labels", type=int, help="Number of labels for the candidate dataset")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name")
    return parser


if __name__ == "__main__":
    parser = data_split_args_parser()
    args = parser.parse_args()
    device = torch.device("cuda")

    model_path = args.model_path
    data_path = args.data_path
    num_labels = args.num_labels
    model_name = args.model_name
    ignore_token = -100

    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))
    # label and id conversion
    labels = []
    for x in df_test["labels"].values:
        labels.extend(x.split(" "))
    unique_labels = set(labels)
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    # model
    if model_name == "bert-base-uncased":
        model = BertModel(model_name=model_name, num_labels=num_labels).to(device)
    elif model_name == "gpt2":
        model = GPTModel(model_name=model_name, num_labels=num_labels).to(device)
    else:
        raise ValueError("model not supported")
    model_weights = torch.load(os.path.join(model_path, "best_FL_global_model.pt"))
    model.load_state_dict(state_dict=model_weights["model"])
    tokenizer = model.tokenizer

    # data
    test_dataset = DataSequence(df_test, labels_to_ids, tokenizer=tokenizer, ignore_token=ignore_token)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=64, shuffle=False)

    # validate
    model.eval()
    with torch.no_grad():
        total_acc_test, total_loss_test, test_total = 0, 0, 0
        test_y_pred, test_y_true = [], []
        for test_data, test_label in test_loader:
            test_label = test_label.to(device)
            test_total += test_label.shape[0]
            mask = test_data["attention_mask"].squeeze(1).to(device)
            input_id = test_data["input_ids"].squeeze(1).to(device)
            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):
                # remove pad tokens
                logits_clean = logits[i][test_label[i] != ignore_token]
                label_clean = test_label[i][test_label[i] != ignore_token]
                # calcluate acc and store prediciton and true labels
                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_test += acc.item()
                test_y_pred.append([ids_to_labels[x.item()] for x in predictions])
                test_y_true.append([ids_to_labels[x.item()] for x in label_clean])
    # metric summary
    summary = classification_report(y_true=test_y_true, y_pred=test_y_pred, zero_division=0)
    print(summary)
