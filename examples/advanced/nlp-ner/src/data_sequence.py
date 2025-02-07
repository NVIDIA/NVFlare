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

import torch


def align_label(
    texts_encoded,
    labels_raw,
    labels_to_ids,
    ignore_token,
):
    # generate label id vector for the network
    # mark the tokens to be ignored
    labels_aligned = []
    # single sentence each time, so always use 0 index
    # get the index mapping from token to word
    # this can be dependent on the specific tokenizer
    word_ids = texts_encoded.word_ids(batch_index=0)
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # set None the ignore tokens
            labels_aligned.append(ignore_token)
        elif word_idx != previous_word_idx:
            # only label the first token of a word
            labels_aligned.append(labels_to_ids[labels_raw[word_idx]])
        else:
            labels_aligned.append(ignore_token)
        previous_word_idx = word_idx
    return labels_aligned


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df, labels_to_ids, tokenizer, ignore_token=-100, max_length=150):
        # Raw texts and corresponding labels
        texts_batch_raw = [i.split(" ") for i in df["text"].values.tolist()]
        labels_batch_raw = [i.split(" ") for i in df["labels"].values.tolist()]
        # Iterate through all cases
        self.texts = []
        self.labels = []
        for batch_idx in range(len(texts_batch_raw)):
            texts_raw = texts_batch_raw[batch_idx]
            labels_raw = labels_batch_raw[batch_idx]
            # Encode texts with tokenizer
            texts_encoded = tokenizer.encode_plus(
                texts_raw,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                is_split_into_words=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            labels_aligned = align_label(texts_encoded, labels_raw, labels_to_ids, ignore_token)
            self.texts.append(texts_encoded)
            self.labels.append(labels_aligned)

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
