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

import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import accuracy_score, f1_score
from transformers import RobertaTokenizer
from utils import hinge_loss


class SST2Metric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("bad", add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode("great", add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()

        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}


class YelpPMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("bad", add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode("great", add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}


class AGNewsMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("World", add_special_tokens=False)[0]: 0,
            tokenizer.encode("Sports", add_special_tokens=False)[0]: 1,
            tokenizer.encode("Business", add_special_tokens=False)[0]: 2,
            tokenizer.encode("Tech", add_special_tokens=False)[0]: 3,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}


class DBPediaMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("Company", add_special_tokens=False)[0]: 0,
            tokenizer.encode("Education", add_special_tokens=False)[0]: 1,
            tokenizer.encode("Artist", add_special_tokens=False)[0]: 2,
            tokenizer.encode("Athlete", add_special_tokens=False)[0]: 3,
            tokenizer.encode("Office", add_special_tokens=False)[0]: 4,
            tokenizer.encode("Transportation", add_special_tokens=False)[0]: 5,
            tokenizer.encode("Building", add_special_tokens=False)[0]: 6,
            tokenizer.encode("Natural", add_special_tokens=False)[0]: 7,
            tokenizer.encode("Village", add_special_tokens=False)[0]: 8,
            tokenizer.encode("Animal", add_special_tokens=False)[0]: 9,
            tokenizer.encode("Plant", add_special_tokens=False)[0]: 10,
            tokenizer.encode("Album", add_special_tokens=False)[0]: 11,
            tokenizer.encode("Film", add_special_tokens=False)[0]: 12,
            tokenizer.encode("Written", add_special_tokens=False)[0]: 13,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}


class MRPCMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("No", add_special_tokens=False)[0]: 0,  # not dumplicate
            tokenizer.encode("Yes", add_special_tokens=False)[0]: 1,  # dumplicate
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        f1 = f1_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"f1": f1, "hinge": hinge_loss, "ce": ce_loss}


class MNLIMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("Yes", add_special_tokens=False)[0]: 0,
            tokenizer.encode("Maybe", add_special_tokens=False)[0]: 1,
            tokenizer.encode("No", add_special_tokens=False)[0]: 2,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )

        target = target.cpu().numpy().tolist()
        for t in target:
            self._target.append(self.label_map[t])

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index].argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        if reset:
            self._target = []
            self._pred = []
        return {"acc": acc}


class RTEMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("Yes", add_special_tokens=False)[0]: 0,
            tokenizer.encode("No", add_special_tokens=False)[0]: 1,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}


class SNLIMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.label_map = {
            tokenizer.encode("Yes", add_special_tokens=False)[0]: 0,
            tokenizer.encode("Maybe", add_special_tokens=False)[0]: 1,
            tokenizer.encode("No", add_special_tokens=False)[0]: 2,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(pred)}."
            )
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor," f"got {type(target)}."
            )
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction="sum").item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {"acc": acc, "hinge": hinge_loss, "ce": ce_loss}
