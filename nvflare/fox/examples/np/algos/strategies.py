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
import random

import numpy as np

from nvflare.fox.api.constants import ContextKey
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fuel.utils.log_utils import get_obj_logger

from .utils import parse_array_def


class NPFedAvgSequential(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        Strategy.__init__(self)
        self.name = "NPFedAvgSequential"
        self.num_rounds = num_rounds
        self.initial_model = parse_array_def(initial_model)
        self.logger = get_obj_logger(self)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self.initial_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model, context)
        return current_model

    def _do_one_round(self, r, current_model, context: Context):
        total = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        n = 0
        for c in context.clients:
            result = c.train(r, current_model)
            self.logger.info(f"[{context.header_str()}] round {r}: got result from client {c.name}: {result}")
            total += result
            n += 1
        return total / n


class NPFedAvgParallel(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = parse_array_def(initial_model)
        self.name = "NPFedAvgParallel"
        self.logger = get_obj_logger(self)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self.initial_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model, context)
            score = self._do_eval(current_model, context)
            self.logger.info(f"[{context.header_str()}]: eval score in round {i}: {score}")
        return current_model

    def _do_eval(self, model, ctx: Context):
        results = all_clients(ctx).evaluate(model)
        total = 0.0
        for n, v in results.items():
            self.logger.info(f"[{ctx.header_str()}]: got eval result from client {n}: {v}")
            total += v
        return total / len(results)

    def _do_one_round(self, r, current_model, ctx: Context):
        total = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        results = all_clients(ctx).train(r, current_model)
        for n, v in results.items():
            self.logger.info(f"[{ctx.header_str()}] round {r}: got group result from client {n}: {v}")
            total += v
        return total / len(results)


class _AggrResult:

    def __init__(self):
        self.total = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        self.count = 0


class NPFedAvgInTime(Strategy):

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = parse_array_def(initial_model)
        self.timeout = timeout
        self.name = "NPFedAvgInTime"
        self.logger = get_obj_logger(self)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self.initial_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model, context)
            score = self._do_eval(current_model, context)
            self.logger.info(f"[{context.header_str()}]: eval score in round {i}: {score}")
        return current_model

    def _do_eval(self, model, ctx: Context):
        results = all_clients(ctx).evaluate(model)
        total = 0.0
        for n, v in results.items():
            self.logger.info(f"[{ctx.header_str()}]: got eval result from client {n}: {v}")
            total += v
        return total / len(results)

    def _do_one_round(self, r, current_model, ctx: Context):
        aggr_result = _AggrResult()
        all_clients(
            ctx,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(r, current_model)

        self.logger.info(
            f"[{ctx.header_str()}] round {r}: aggr result from {aggr_result.count} clients: {aggr_result.total}"
        )
        if aggr_result.count == 0:
            return None
        else:
            return aggr_result.total / aggr_result.count

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] got train result from {context.caller} {result}")
        aggr_result.total += result
        aggr_result.count += 1
        return None


class NPCyclic(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = parse_array_def(initial_model)
        self.logger = get_obj_logger(self)

    def execute(self, context: Context):
        current_model = context.get_prop(ContextKey.INPUT, self.initial_model)
        for current_round in range(self.num_rounds):
            current_model = self._do_one_round(current_round, current_model, context)
        self.logger.info(f"[{context.header_str()}] final result: {current_model}")
        return current_model

    def _do_one_round(self, current_round, current_model, ctx: Context):
        random.shuffle(ctx.clients)
        for c in ctx.clients:
            current_model = c.train(current_round, current_model)
            self.logger.info(f"[{ctx.header_str()}] result from {c.name}: {current_model}")
        return current_model
