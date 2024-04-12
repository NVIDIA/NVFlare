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


import copy
import os

import numpy as np
import torch
from fastNLP import DataSet, DataSetIter, SequentialSampler, Tester
from models.modeling_roberta import RobertaForMaskedLM
from sklearn.metrics import f1_score
from transformers import RobertaConfig, RobertaTokenizer
from utils import hinge_loss


class LMForwardAPI:
    def __init__(self, args, train_data=None, dev_data=None, init_prompt_path=None, baseAPI=True):
        model_name = args.model_name
        from metrics.metrics import (
            AGNewsMetric,
            DBPediaMetric,
            MRPCMetric,
            RTEMetric,
            SNLIMetric,
            SST2Metric,
            YelpPMetric,
        )

        task_name = args.task_name
        if task_name in ["sst2", "yelpp", "rte", "mrpc", "chnsent", "lcqmc", "bq"]:
            self.num_labels = 2
        elif task_name in ["snli", "cmnli", "ocnli"]:
            self.num_labels = 3
        elif task_name in ["agnews", "ccpm", "c3"]:
            self.num_labels = 4
        elif task_name in ["amazon"]:
            self.num_labels = 5
        elif task_name in ["thucnews"]:
            self.num_labels = 10
        elif task_name in ["dbpedia", "tnews"]:
            self.num_labels = 14
        else:
            raise ValueError
        n_prompt_tokens = args.n_prompt_tokens
        intrinsic_dim = args.intrinsic_dim

        sigma = args.sigma
        alpha = args.alpha

        self.args = args

        device = args.device
        random_proj = args.random_proj
        loss_type = args.loss_type
        print_every = args.print_every
        eval_every = args.eval_every
        cat_or_add = args.cat_or_add

        inference_framework = args.inference_framework
        onnx_model_path = args.onnx_model_path

        self.model_name = args.model_name
        self.parallel = args.parallel
        self.n_prompt_tokens = args.n_prompt_tokens
        self.batch_size = args.batch_size
        self.device = args.device

        if inference_framework not in ["pt", "ort"]:
            raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
        if inference_framework == "ort":
            assert onnx_model_path is not None, "Path to onnx model is required, got None instead."
            assert os.path.exists(onnx_model_path), f"In valid onnx model path `{onnx_model_path}`"

        self.train_data = train_data
        self.dev_data = dev_data
        self.train_data_aux = None
        self.config = RobertaConfig.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(
            model_name,
            config=self.config,
            n_prompt_tokens=n_prompt_tokens,
            inference_framework=inference_framework,
            onnx_model_path=onnx_model_path,
        )
        self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))

        if inference_framework == "ort":
            self.model.roberta = None
        if cat_or_add == "cat":
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print("Initialize prompt embedding from {}".format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print("Initial prompt embedding not found. Initialize to zero embedding.")
                self.init_prompt = torch.zeros(n_prompt_tokens * self.config.hidden_size)
            print("Shape of initial prompt embedding: {}".format(self.init_prompt.shape))
        else:
            # self.model.set_concat_prompt(False)
            self.init_prompt = None

        if args.init_score_path is not None:
            if args.llama_causal:
                raise ValueError("You cannot initilize a score layer for a causal model")
            score_state = self.model.score.state_dict()
            score_state["weight"] = torch.load(args.init_score_path)
            self.model.score.load_state_dict(score_state)
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if random_proj == "normal":
            # calculate std for normal distribution
            embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()

            # embedding = embedding[1000: 2000]
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print("[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}".format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        if task_name == "sst2":
            self.metric = SST2Metric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "SST2Metric"
        elif task_name == "agnews":
            self.metric = AGNewsMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "AGNewsMetric"
        elif task_name == "yelpp":
            self.metric = YelpPMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "YelpPMetric"
        elif task_name == "dbpedia":
            self.metric = DBPediaMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "DBPediaMetric"
        elif task_name == "rte":
            self.metric = RTEMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "RTEMetric"
        elif task_name == "mrpc":
            self.metric = MRPCMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "f1"
            self.metric_name = "MRPCMetric"
        elif task_name == "snli":
            self.metric = SNLIMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "SNLIMetric"
        elif task_name == "chnsent":
            self.metric = ChnSentMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "ChnSentMetric"
        elif task_name == "thucnews":
            self.metric = THUCNewsMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "THUCNewsMetric"
        elif task_name == "lcqmc":
            self.metric = LCQMCMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "LCQMCMetric"
        elif task_name == "cmnli":
            self.metric = CMNLIMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "CMNLIMetric"
        elif task_name == "ocnli":
            self.metric = OCNLIMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "OCNLIMetric"
        elif task_name == "amazon":
            self.metric = AmazonMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "AmazonMetric"
        elif task_name == "bq":
            self.metric = BQMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "BQMetric"
        elif task_name == "ccpm":
            self.metric = CCPMMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "CCPMMetric"
        elif task_name == "tnews":
            self.metric = TNewsMetric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "TNewsMetric"
        elif task_name == "c3":
            self.metric = C3Metric(target="labels", pred="logits", tokenizer=self.tokenizer)
            self.metric_key = "acc"
            self.metric_name = "C3Metric"
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def convert_pred(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        if self.args.model_name not in ["llama2"] or self.args.llama_causal:
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)
        return pred, converted_target

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        if self.args.model_name not in ["llama2"] or self.args.llama_causal:
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == "acc":
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == "f1":
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f"[Metric] Only support [acc, f1], got {self.metric_key} instead.")

        if self.loss_type == "hinge":
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction="sum").item() / len(target)
        elif self.loss_type == "ce":
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == "perf":
            loss = -1 * perf
        else:
            raise KeyError(f"[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.")

        return loss, perf

    def set_dataset(self, train_data, dev_data, train_data_aux=None):
        self.train_data, self.dev_data = train_data, dev_data
        if train_data_aux is not None:
            self.train_data_aux = train_data_aux

    def load_client_record(self, record):
        self.best_train_perf = record["best_train_perf"]
        self.best_dev_perf = record["best_dev_perf"]
        self.best_prompt = record["best_prompt"]
        self.num_call = record["num_call"]

    def client_record(self):
        record = {}
        record["best_train_perf"] = copy.deepcopy(self.best_train_perf)
        record["best_dev_perf"] = copy.deepcopy(self.best_dev_perf)
        record["best_prompt"] = copy.deepcopy(self.best_prompt)
        record["num_call"] = copy.deepcopy(self.num_call)
        return record

    # def inference(self, model, data):
    #     for k, v in data.items():
    #         data[k] = v.to(self.device)
    #     with torch.no_grad():
    #         if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #                 decoder_input_ids=data['decoder_input_ids'],
    #                 decoder_attention_mask=data['decoder_attention_mask'],
    #             )['logits']
    #         elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #             )['logits']
    #         else:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #                 mask_pos=data['mask_pos'],
    #             )['logits']

    #     target = data['labels']
    #     label_map = self.metric.label_map

    #     converted_target = target.clone()
    #     for key, val in label_map.items():
    #         converted_target[target == key] = val
    #     interest_index = list(label_map.keys())
    #     logits = logits[:, interest_index]
    #     pred = logits.argmax(dim=-1)
    #     return pred, converted_target

    def eval(self, prompt_embedding=None, test_data=None, return_pred=False):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz_dev = len(self.dev_data["input_ids"])
            bsz_train = len(self.train_data["input_ids"])
            bsz = bsz_train if bsz_train > bsz_dev else bsz_dev
        else:
            bsz = self.batch_size  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            if self.args.norm_prompt:
                for i in range(len(prompt_embedding)):
                    if np.linalg.norm(prompt_embedding[i]) > self.args.prompt_norm_threshold:
                        prompt_embedding[i] = (
                            prompt_embedding[i] / np.linalg.norm(prompt_embedding[i]) * self.args.prompt_norm_threshold
                        )
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(self.train_data["input_ids"])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            if self.args.norm_prompt:
                if np.linalg.norm(prompt_embedding) > self.args.prompt_norm_threshold:
                    prompt_embedding = (
                        prompt_embedding / np.linalg.norm(prompt_embedding) * self.args.prompt_norm_threshold
                    )
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )

        self.model.set_prompt_embedding(prompt_embedding)

        if return_pred is True:
            if self.parallel:  # if we have multiple queries, use the one that achieves minimal loss
                self.model.set_prompt_embedding(prompt_embedding)
            for k, v in self.dev_data.items():
                self.dev_data[k] = v.to(self.device)
            with torch.no_grad():
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                        decoder_input_ids=self.dev_data["decoder_input_ids"],
                        decoder_attention_mask=self.dev_data["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                        mask_pos=self.dev_data["mask_pos"],
                    )["logits"]
            pred, labels = self.convert_pred(logits, self.dev_data["labels"])
            return pred, labels

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError("Provide a single prompt embedding for testing.")

            test_tester = Tester(
                data=test_data,
                model=self.model,
                metrics=self.metric,
                batch_size=self.batch_size,
                num_workers=1,
                device=self.device,
                use_tqdm=False,
            )
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in self.train_data.items():
                self.train_data[k] = v.to(self.device)
            with torch.no_grad():
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=self.train_data["input_ids"],
                        attention_mask=self.train_data["attention_mask"],
                        decoder_input_ids=self.train_data["decoder_input_ids"],
                        decoder_attention_mask=self.train_data["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=self.train_data["input_ids"],
                        attention_mask=self.train_data["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=self.train_data["input_ids"],
                        attention_mask=self.train_data["attention_mask"],
                        mask_pos=self.train_data["mask_pos"],
                    )["logits"]

            if self.parallel:  # we have multiple queries
                all_losses, all_perfs = [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz : i * bsz + bsz]
                    tmp_target = self.train_data["labels"][i * bsz : i * bsz + bsz]
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, self.train_data["labels"])
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            # if self.num_call % self.print_every == 0:
            #     print(
            #         '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
            #             self.num_call,
            #             round(float(loss), 4),
            #             round(float(perf), 4),
            #             round(float(self.best_train_perf), 4)))

            # if self.num_call % self.eval_every == 0:
            #     print('********* Evaluated on dev set *********')
            #     if self.parallel:  # if we have multiple queries, use the one that achieves minimal loss
            #         self.model.set_prompt_embedding(prompt_embedding)
            #     for k, v in self.dev_data.items():
            #         self.dev_data[k] = v.to(self.device)
            #     with torch.no_grad():
            #         if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #                 decoder_input_ids=self.dev_data['decoder_input_ids'],
            #                 decoder_attention_mask=self.dev_data['decoder_attention_mask'],
            #             )['logits']
            #         elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama2']:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #             )['logits']
            #         else:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #                 mask_pos=self.dev_data['mask_pos'],
            #             )['logits']

            #     dev_loss, dev_perf = self.calc_metric(logits, self.dev_data['labels'])
            #     # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
            #     if dev_perf > self.best_dev_perf:
            #         self.best_dev_perf = dev_perf
            #         # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
            #         self.best_prompt = copy.deepcopy(tmp_prompt)
            #     # if self.save_path is not None:
            #     #     with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
            #     #         fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
            #     print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            #         round(float(dev_loss), 4),
            #         round(float(dev_perf), 4),
            #         round(float(self.best_dev_perf), 4)))
            #     print('********* Done *********')
            if self.parallel:
                return all_losses
            else:
                return loss

    def eval_perturb(self, prompt_embedding=None, test_data=None, return_pred=False):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz_dev = len(self.dev_data["input_ids"])
            bsz_train = len(self.train_data_aux["input_ids"])
            bsz = bsz_train if bsz_train > bsz_dev else bsz_dev
        else:
            bsz = self.batch_size  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            if self.args.norm_prompt:
                for i in range(len(prompt_embedding)):
                    if np.linalg.norm(prompt_embedding[i]) > self.args.prompt_norm_threshold:
                        prompt_embedding[i] = (
                            prompt_embedding[i] / np.linalg.norm(prompt_embedding[i]) * self.args.prompt_norm_threshold
                        )
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(self.train_data_aux["input_ids"])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            if self.args.norm_prompt:
                if np.linalg.norm(prompt_embedding) > self.args.prompt_norm_threshold:
                    prompt_embedding = (
                        prompt_embedding / np.linalg.norm(prompt_embedding) * self.args.prompt_norm_threshold
                    )
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )

        self.model.set_prompt_embedding(prompt_embedding)

        if return_pred is True:
            if self.parallel:  # if we have multiple queries, use the one that achieves minimal loss
                self.model.set_prompt_embedding(prompt_embedding)
            for k, v in self.dev_data.items():
                self.dev_data[k] = v.to(self.device)
            with torch.no_grad():
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                        decoder_input_ids=self.dev_data["decoder_input_ids"],
                        decoder_attention_mask=self.dev_data["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=self.dev_data["input_ids"],
                        attention_mask=self.dev_data["attention_mask"],
                        mask_pos=self.dev_data["mask_pos"],
                    )["logits"]
            pred, labels = self.convert_pred(logits, self.dev_data["labels"])
            return pred, labels

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError("Provide a single prompt embedding for testing.")

            test_tester = Tester(
                data=test_data,
                model=self.model,
                metrics=self.metric,
                batch_size=self.batch_size,
                num_workers=1,
                device=self.device,
                use_tqdm=False,
            )
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in self.train_data_aux.items():
                self.train_data_aux[k] = v.to(self.device)
            with torch.no_grad():
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=self.train_data_aux["input_ids"],
                        attention_mask=self.train_data_aux["attention_mask"],
                        decoder_input_ids=self.train_data_aux["decoder_input_ids"],
                        decoder_attention_mask=self.train_data_aux["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=self.train_data_aux["input_ids"],
                        attention_mask=self.train_data_aux["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=self.train_data_aux["input_ids"],
                        attention_mask=self.train_data_aux["attention_mask"],
                        mask_pos=self.train_data_aux["mask_pos"],
                    )["logits"]

            if self.parallel:  # we have multiple queries
                all_losses, all_perfs = [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz : i * bsz + bsz]
                    tmp_target = self.train_data_aux["labels"][i * bsz : i * bsz + bsz]
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, self.train_data_aux["labels"])
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            # if self.num_call % self.print_every == 0:
            #     print(
            #         '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
            #             self.num_call,
            #             round(float(loss), 4),
            #             round(float(perf), 4),
            #             round(float(self.best_train_perf), 4)))

            # if self.num_call % self.eval_every == 0:
            #     print('********* Evaluated on dev set *********')
            #     if self.parallel:  # if we have multiple queries, use the one that achieves minimal loss
            #         self.model.set_prompt_embedding(prompt_embedding)
            #     for k, v in self.dev_data.items():
            #         self.dev_data[k] = v.to(self.device)
            #     with torch.no_grad():
            #         if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #                 decoder_input_ids=self.dev_data['decoder_input_ids'],
            #                 decoder_attention_mask=self.dev_data['decoder_attention_mask'],
            #             )['logits']
            #         elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama2']:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #             )['logits']
            #         else:
            #             logits = self.model(
            #                 input_ids=self.dev_data['input_ids'],
            #                 attention_mask=self.dev_data['attention_mask'],
            #                 mask_pos=self.dev_data['mask_pos'],
            #             )['logits']

            #     dev_loss, dev_perf = self.calc_metric(logits, self.dev_data['labels'])
            #     # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
            #     if dev_perf > self.best_dev_perf:
            #         self.best_dev_perf = dev_perf
            #         # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
            #         self.best_prompt = copy.deepcopy(tmp_prompt)
            #     # if self.save_path is not None:
            #     #     with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
            #     #         fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
            #     print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            #         round(float(dev_loss), 4),
            #         round(float(dev_perf), 4),
            #         round(float(self.best_dev_perf), 4)))
            #     print('********* Done *********')
            if self.parallel:
                return all_losses
            else:
                return loss

    def eval_multi_batch(self, prompt_embedding=None, test_data=None, return_pred=False):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        bsz = self.batch_size  # for test data
        if isinstance(prompt_embedding, list):  # multiple queries
            if self.args.norm_prompt:
                for i in range(len(prompt_embedding)):
                    if np.linalg.norm(prompt_embedding[i]) > self.args.prompt_norm_threshold:
                        prompt_embedding[i] = (
                            prompt_embedding[i] / np.linalg.norm(prompt_embedding[i]) * self.args.prompt_norm_threshold
                        )
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(self.train_data["input_ids"])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            if self.args.norm_prompt:
                if np.linalg.norm(prompt_embedding) > self.args.prompt_norm_threshold:
                    prompt_embedding = (
                        prompt_embedding / np.linalg.norm(prompt_embedding) * self.args.prompt_norm_threshold
                    )
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )

        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError("Provide a single prompt embedding for testing.")

            test_tester = Tester(
                data=test_data,
                model=self.model,
                metrics=self.metric,
                batch_size=self.batch_size,
                num_workers=1,
                device=self.device,
                use_tqdm=False,
            )
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            dataloader_train = DataSetIter(self.train_data, batch_size=self.batch_size, sampler=SequentialSampler())
            loss_list = []
            for train_data, train_label in dataloader_train:
                for k, v in train_data.items():
                    train_data[k] = v.to(self.device)
                for k, v in train_label.items():
                    train_label[k] = v.to(self.device)
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                        decoder_input_ids=train_data["decoder_input_ids"],
                        decoder_attention_mask=train_data["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                        mask_pos=train_data["mask_pos"],
                    )["logits"]

                loss, perf = self.calc_metric(logits, train_label["labels"])
                loss_list.append(loss)

            return np.average(loss_list)

    def eval_perturb_multi_batch(self, prompt_embedding=None, test_data=None, return_pred=False):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        bsz = self.batch_size  # for test data
        if isinstance(prompt_embedding, list):  # multiple queries
            if self.args.norm_prompt:
                for i in range(len(prompt_embedding)):
                    if np.linalg.norm(prompt_embedding[i]) > self.args.prompt_norm_threshold:
                        prompt_embedding[i] = (
                            prompt_embedding[i] / np.linalg.norm(prompt_embedding[i]) * self.args.prompt_norm_threshold
                        )
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(self.train_data_aux["input_ids"])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            if self.args.norm_prompt:
                if np.linalg.norm(prompt_embedding) > self.args.prompt_norm_threshold:
                    prompt_embedding = (
                        prompt_embedding / np.linalg.norm(prompt_embedding) * self.args.prompt_norm_threshold
                    )
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )

        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError("Provide a single prompt embedding for testing.")

            test_tester = Tester(
                data=test_data,
                model=self.model,
                metrics=self.metric,
                batch_size=self.batch_size,
                num_workers=1,
                device=self.device,
                use_tqdm=False,
            )
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            dataloader_train = DataSetIter(self.train_data_aux, batch_size=bsz, sampler=SequentialSampler())
            loss_list = []
            for train_data, train_label in dataloader_train:
                for k, v in train_data.items():
                    train_data[k] = v.to(self.device)
                for k, v in train_label.items():
                    train_label[k] = v.to(self.device)
                if self.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                        decoder_input_ids=train_data["decoder_input_ids"],
                        decoder_attention_mask=train_data["decoder_attention_mask"],
                    )["logits"]
                elif self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                    )["logits"]
                else:
                    logits = self.model(
                        input_ids=train_data["input_ids"],
                        attention_mask=train_data["attention_mask"],
                        mask_pos=train_data["mask_pos"],
                    )["logits"]

                loss, perf = self.calc_metric(logits, train_label["labels"])
                loss_list.append(loss)

            return np.average(loss_list)


class ClientLMForwardAPI(LMForwardAPI):
    def __init__(self, args, train_data=None, dev_data=None, init_prompt_path=None, baseAPI=None):
        super().__init__(args, train_data, dev_data, init_prompt_path)
        if not isinstance(baseAPI, LMForwardAPI):
            raise ValueError("Please provide a base API to initialize API for the clients")
        self.model = baseAPI.model
        self.tokenizer = baseAPI.tokenizer
        self.config = baseAPI.config
        self.metric = baseAPI.metric
        self.metric_key = baseAPI.metric_key
        self.metric_name = baseAPI.metric_name
        self.linear = baseAPI.linear
