# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import re

import torch


def align_label(
    tokenized_inputs,
    origional_text,
    labels,
    labels_to_ids,
    pad_token,
    tokenizer=None,
):
    label_ids = []
    origional_text = origional_text.split(" ")

    orig_labels_i = 0
    partially_mathced = False
    sub_str = str()
    for token_id in tokenized_inputs["input_ids"][0]:
        token_id = token_id.numpy().item()
        cur_str = tokenizer.convert_ids_to_tokens(token_id).lower()
        if (
            (token_id == tokenizer.pad_token_id)
            or (token_id == tokenizer.cls_token_id)
            or (token_id == tokenizer.sep_token_id)
        ):

            label_ids.append(pad_token)

        elif (
            (not partially_mathced)
            and origional_text[orig_labels_i].lower().startswith(cur_str)
            and origional_text[orig_labels_i].lower() != cur_str
        ):

            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = True
            sub_str += cur_str

        elif (not partially_mathced) and origional_text[orig_labels_i].lower() == cur_str:
            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = False

        else:
            label_ids.append(pad_token)
            sub_str += re.sub("#+", "", cur_str)
            if sub_str == origional_text[orig_labels_i - 1].lower():
                partially_mathced = False
                sub_str = ""

    return label_ids


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df, labels_to_ids, tokenizer, pad_token=-100, max_length=150):
        lb = [i.split(" ") for i in df["labels"].values.tolist()]
        txt = df["text"].values.tolist()
        self.texts = [
            tokenizer.encode_plus(
                str(i),
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            for i in txt
        ]
        self.labels = [
            align_label(t, tt, l, labels_to_ids=labels_to_ids, pad_token=pad_token, tokenizer=tokenizer)
            for t, tt, l in zip(self.texts, txt, lb)
        ]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
