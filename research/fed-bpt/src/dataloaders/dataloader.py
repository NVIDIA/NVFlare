# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Part of this code is adopted from BBT (https://github.com/txsun1997/Black-Box-Tuning)

# MIT License
#
# Copyright (c) 2022 Tianxiang Sun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial

import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle, Loader
from transformers import RobertaTokenizer


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch["input_text"])
    target_encodings = tokenizer.batch_encode_plus(example_batch["target_text"], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings["input_ids"]:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "mask_pos": mask_pos,
        "labels": target_encodings["input_ids"],
    }

    return encodings


class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s . It was %s ." % (prompt, example["sentence"], self.tokenizer.mask_token)
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s . It was %s ." % (example["sentence"], self.tokenizer.mask_token)
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("glue", "sst2", split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print("Example in {} set:".format(split))
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s . It was %s ." % (
                prompt,
                example["text"].replace("\\n", " "),
                self.tokenizer.mask_token,
            )
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s . It was %s ." % (
                example["text"].replace("\\n", " "),
                self.tokenizer.mask_token,
            )
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("yelp_polarity", "plain_text", split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {0: "World", 1: "Sports", 2: "Business", 3: "Tech"}

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s News: %s" % (prompt, self.tokenizer.mask_token, example["text"])
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s News: %s" % (self.tokenizer.mask_token, example["text"])
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("ag_news", "default", split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class DBPediaLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Company",
            1: "Education",
            2: "Artist",
            3: "Athlete",
            4: "Office",
            5: "Transportation",
            6: "Building",
            7: "Natural",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "Written",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s [ Category: %s ] %s" % (
                prompt,
                self.tokenizer.mask_token,
                example["content"].strip(),
            )
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "[ Category: %s ] %s" % (self.tokenizer.mask_token, example["content"].strip())
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("dbpedia_14", split=split)
        # dataset = datasets.load_dataset('./data/dbpedia.py', split=split)  # if you cannot reach the source of dbpedia, try this
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s ? %s , %s" % (
                prompt,
                example["sentence1"],
                self.tokenizer.mask_token,
                example["sentence2"],
            )
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s ? %s , %s" % (
                example["sentence1"],
                self.tokenizer.mask_token,
                example["sentence2"],
            )
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("glue", "mrpc", split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s ? %s , %s" % (
                prompt,
                example["sentence1"],
                self.tokenizer.mask_token,
                example["sentence2"],
            )
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s ? %s , %s" % (
                example["sentence1"],
                self.tokenizer.mask_token,
                example["sentence2"],
            )
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("glue", "rte", split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example["input_text"] = "%s . %s ? %s , %s" % (
                prompt,
                example["premise"],
                self.tokenizer.mask_token,
                example["hypothesis"],
            )
            example["target_text"] = self.label2text[example["label"]]
        else:
            example["input_text"] = "%s ? %s , %s" % (
                example["premise"],
                self.tokenizer.mask_token,
                example["hypothesis"],
            )
            example["target_text"] = self.label2text[example["label"]]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset("snli", split=split)
        dataset = dataset.filter(lambda example: example["label"] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(
            partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False
        )
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
