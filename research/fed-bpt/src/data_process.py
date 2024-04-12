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

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import random

import numpy as np
import torch
from cvxopt import matrix, solvers
from fastNLP import DataSet, cache_results
from numpy.random import RandomState
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BartTokenizer,
    BertTokenizer,
    ElectraTokenizer,
    GPT2Tokenizer,
    RobertaTokenizer,
    T5Tokenizer,
)

cache_fn = None


class data_processor:
    def __init__(self, args) -> None:
        # below are free hyper-params
        self.model_name = args.model_name
        if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
            from dataloaders.dataloader_t5 import (
                AGNewsLoader,
                DBPediaLoader,
                MRPCLoader,
                RTELoader,
                SNLILoader,
                SST2Loader,
                YelpPLoader,
            )
        elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            from dataloaders.dataloader_gpt import (
                AGNewsLoader,
                DBPediaLoader,
                MRPCLoader,
                RTELoader,
                SNLILoader,
                SST2Loader,
                YelpPLoader,
            )
        elif self.model_name in ["fnlp/cpt-large"]:
            from dataloaders.dataloader_cpt import (
                AmazonLoader,
                BQLoader,
                C3Loader,
                CCPMLoader,
                ChnSentLoader,
                CMNLILoader,
                LCQMCLoader,
                OCNLILoader,
                THUCNewsLoader,
                TNewsLoader,
            )
        elif self.model_name in ["llama2"]:
            from dataloaders.dataloader_llama import (
                AGNewsLoader,
                DBPediaLoader,
                MRPCLoader,
                RTELoader,
                SNLILoader,
                SST2Loader,
                YelpPLoader,
            )
        else:
            from dataloaders.dataloader import (
                AGNewsLoader,
                DBPediaLoader,
                MRPCLoader,
                RTELoader,
                SNLILoader,
                SST2Loader,
                YelpPLoader,
            )

        self.task_name = args.task_name
        self.n_prompt_tokens = args.n_prompt_tokens

        self.seed = args.seed

        # if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
        #     args.cat_or_add = 'cat'
        self.cat_or_add = args.cat_or_add

        if self.task_name in ["sst2", "yelpp", "rte", "mrpc", "chnsent", "lcqmc", "bq"]:
            num_labels = 2
        elif self.task_name in ["snli", "cmnli", "ocnli"]:
            num_labels = 3
        elif self.task_name in ["agnews", "ccpm", "c3"]:
            num_labels = 4
        elif self.task_name in ["amazon"]:
            num_labels = 5
        elif self.task_name in ["thucnews"]:
            num_labels = 10
        elif self.task_name in ["dbpedia", "tnews"]:
            num_labels = 14
        else:
            raise ValueError

        # log_dir = './logs'
        # fitlog.set_log_dir(log_dir)
        # fitlog.commit(__file__, fit_msg=save_path)
        # fitlog.add_hyper(args)
        # fitlog.add_hyper_in_file(__file__)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.model_name in ["roberta-base", "roberta-large"]:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["bert-base-uncased", "bert-large-uncased", "fnlp/cpt-large"]:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["google/electra-base-generator", "google/electra-large-generator"]:
            self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["facebook/bart-base", "facebook/bart-large"]:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        elif self.model_name in ["llama2"]:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            raise NotImplementedError

        global cache_fn
        cache_fn = (
            f"caches/data_{self.model_name.replace('/', '-')}_{self.task_name}_{self.n_prompt_tokens}_{self.seed}.pt"
        )

        if self.model_name not in ["fnlp/cpt-large"]:
            self.DataLoader = {
                "sst2": SST2Loader,
                "agnews": AGNewsLoader,
                "yelpp": YelpPLoader,
                "dbpedia": DBPediaLoader,
                "rte": RTELoader,
                "mrpc": MRPCLoader,
                "snli": SNLILoader,
            }
        else:
            self.DataLoader = {
                "chnsent": ChnSentLoader,
                "thucnews": THUCNewsLoader,
                "lcqmc": LCQMCLoader,
                "cmnli": CMNLILoader,
                "ocnli": OCNLILoader,
                "amazon": AmazonLoader,
                "bq": BQLoader,
                "ccpm": CCPMLoader,
                "tnews": TNewsLoader,
                "c3": C3Loader,
            }

    # @cache_results(cache_fn, _refresh=False)
    def get_data(self):
        if self.task_name in ["agnews", "yelpp", "dbpedia", "snli"]:
            splits = ["train", "test"]
        else:  # for datasets without test set, we use dev set
            splits = ["train", "validation"]
        if self.cat_or_add == "cat":
            data_bundle = self.DataLoader[self.task_name](tokenizer=self.tokenizer, n_prompt_tokens=0).my_load(splits)
        else:
            data_bundle = self.DataLoader[self.task_name](
                tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens
            ).my_load(splits)
        return data_bundle


def construct_true_few_shot_data(args, train_data, k_shot):
    train_label_count = {}
    dev_label_count = {}
    new_train_data = DataSet()
    new_dev_data = DataSet()
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    if k_shot < 0:
        idxs_train = np.random.choice(len(train_data), int(len(train_data) * 0.9), replace=False)
        idxs_dev = list(set(range(len(train_data))) - set(idxs_train))
        new_train_data = train_data[idxs_train.tolist()]
        new_dev_data = train_data[np.array(idxs_dev).tolist()]

    else:
        for index in all_indices:
            label = train_data[index]["labels"]
            if label < 0:
                continue

            if label not in train_label_count:
                train_label_count[label] = 0
            if label not in dev_label_count:
                dev_label_count[label] = 0

            if train_label_count[label] < k_shot:
                new_train_data.append(train_data[index])
                train_label_count[label] += 1
            elif dev_label_count[label] < k_shot:
                new_dev_data.append(train_data[index])
                dev_label_count[label] += 1

    if args.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
        new_train_data.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        new_dev_data.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
    elif args.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
        new_train_data.set_input("input_ids", "attention_mask")
        new_dev_data.set_input("input_ids", "attention_mask")
    else:
        new_train_data.set_input("input_ids", "attention_mask", "mask_pos")
        new_dev_data.set_input("input_ids", "attention_mask", "mask_pos")

    new_train_data.set_target("labels")
    new_dev_data.set_target("labels")
    return new_train_data, new_dev_data


def split_data(args, train_data, dev_data):
    train_data_idxs = [i for i in range(len(train_data))]
    dev_data_idxs = [i for i in range(len(dev_data))]
    user_dict_train, user_dict_dev = {}, {}
    num_items_train = int(len(train_data) / args.num_users)
    num_items_dev = int(len(dev_data) / args.num_users)

    if args.iid == 1:
        for i in range(args.num_users):
            user_dict_train[i] = set(np.random.choice(train_data_idxs, num_items_train, replace=False))
            user_dict_dev[i] = set(np.random.choice(dev_data_idxs, num_items_dev, replace=False))
            train_data_idxs = list(set(train_data_idxs) - user_dict_train[i])
            dev_data_idxs = list(set(dev_data_idxs) - user_dict_dev[i])

    if args.iid == 0:
        rs = RandomState(args.seed)
        user_dict_train, _ = Dirichlet_noniid(train_data, args.num_users, args.alpha_dir, rs)
        for i in range(args.num_users):
            user_dict_dev[i] = set(np.random.choice(dev_data_idxs, num_items_dev, replace=False))
            dev_data_idxs = list(set(dev_data_idxs) - user_dict_dev[i])

    return user_dict_train, user_dict_dev


def Dirichlet_noniid(dataset, num_users, alpha, rs):
    """
    Sample dataset with dirichlet distribution and concentration parameter alpha
    """
    # img_num_per_client = len(dataset)//num_users
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset["labels"])
    classes = np.unique(labels)
    num_classes = len(classes)
    labels_idxs = []
    prior_class_distribution = np.zeros(num_classes)
    b = np.zeros(num_classes)
    for i in range(num_classes):
        labels_idxs.append(idxs[labels == classes[i]])
        prior_class_distribution[i] = len(labels_idxs[i]) / len(dataset)
        b[i] = len(labels_idxs[i])

    data_ratio = np.zeros([num_classes, num_users])

    if isinstance(alpha, list):
        for i in range(num_users):
            data_ratio[:, i] = rs.dirichlet(prior_class_distribution * alpha[i])
    else:
        data_ratio = np.transpose(rs.dirichlet(prior_class_distribution * alpha, size=num_users))
    # data_ratio = data_ratio/np.sum(data_ratio,axis=1,keepdims=True)
    # Client_DataSize = len(dataset)//num_users*np.ones([num_users,1],dtype=np.int64)
    print(f"Class_distribution {prior_class_distribution}. Data_ratio {data_ratio}")
    A = matrix(data_ratio)
    b = matrix(b)
    G = matrix(-np.eye(num_users))
    h = matrix(np.zeros([num_users, 1]))
    P = matrix(np.eye(num_users))
    q = matrix(np.zeros([num_users, 1]))
    try:
        results = solvers.qp(P, q, G, h, A, b)
        Client_DataSize = np.array(results["x"])
        Data_Division = data_ratio * np.transpose(Client_DataSize)
    except ValueError:
        prior_user_distribution = np.array([1 / num_users for _ in range(num_users)])
        data_ratio = rs.dirichlet(prior_user_distribution * alpha, size=num_classes)
        Class_DataSize = np.array([int(len(dataset) / num_classes) for _ in range(num_classes)])
        Data_Division = (data_ratio.T * Class_DataSize).T
    # print(Client_DataSize)
    print(Data_Division)
    print(np.sum(Data_Division, axis=0))
    print(np.sum(Data_Division, axis=1))
    rest = []
    for label in range(num_classes):
        for client in range(num_users):
            data_idx = rs.choice(labels_idxs[label], int(Data_Division[label, client]), replace=False)
            dict_users[client] = np.concatenate([dict_users[client], data_idx], 0)
            labels_idxs[label] = list(set(labels_idxs[label]) - set(data_idx))
        rest = rest + labels_idxs[label]

    rest_clients = rs.choice(range(num_users), len(rest), replace=True)

    for n, user in enumerate(rest_clients):
        dict_users[user] = np.append(dict_users[user], rest[n])

    for user in range(num_users):
        rs.shuffle(dict_users[user])
    return dict_users, data_ratio


def perturb_dataset(args, dataset, config):
    pert_dataset = copy.deepcopy(dataset)
    if isinstance(dataset, dict):
        preserve_mask = torch.ones_like(dataset["input_ids"])
        random_text = torch.randint_like(dataset["input_ids"], 0, config.vocab_size)
        replace_mask = torch.bernoulli(args.perturb_rate * dataset["attention_mask"]).long()
        preserve_mask -= replace_mask
        pert_dataset["input_ids"] = pert_dataset["input_ids"] * preserve_mask + random_text * replace_mask
        return pert_dataset
    else:
        input_content = torch.tensor(pert_dataset["input_ids"].get(range(len(dataset))))
        preserve_mask = torch.ones_like(input_content)
        random_text = torch.randint_like(input_content, 0, config.vocab_size)
        replace_mask = torch.bernoulli(
            args.perturb_rate * torch.tensor(dataset["attention_mask"].get(range(len(dataset))))
        ).long()
        preserve_mask -= replace_mask
        pert_dataset["input_ids"].content = (input_content * preserve_mask + random_text * replace_mask).tolist()
        pert_dataset["attention_mask"].content = torch.tensor(
            pert_dataset["attention_mask"].get(range(len(dataset)))
        ).tolist()
        pert_dataset["mask_pos"].content = torch.tensor(pert_dataset["mask_pos"].get(range(len(dataset)))).tolist()
        pert_dataset["labels"].content = torch.tensor(pert_dataset["labels"].get(range(len(dataset)))).tolist()
        return pert_dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[item]
