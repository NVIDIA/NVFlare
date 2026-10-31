# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""FedMoSA server workflow.

Per round the server
  * aggregates the expert of modality m (``experts.<idx(m)>``) only over clients holding m, weighted by each
    client's number of m samples (modality-restricted aggregation);
  * aggregates every other exchanged parameter (shared adapters, mask decoder) with weights
    proportional to |D_k| * |M_k| (modality-balanced aggregation);
  * aggregates the shared-adapter prototypes with the same balanced weights and sends them back for the
    prototype-alignment loss.

Clients send ``meta = {"mod_counts": {modality: n}, "prototypes": [{"attn_shared": t, "mlp_shared": t}, ...]}``.
"""

import re

import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController

EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


class FedMoSAController(ModelController):
    def __init__(self, *args, num_clients: int = 2, num_rounds: int = 10, modalities=("MR", "CT"), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.modalities = list(modalities)

    def run(self):
        self.info("Start FedMoSA.")
        model = self.load_model()
        for rnd in range(self.num_rounds):
            self.info(f"Round {rnd}")
            model.current_round = rnd
            clients = self.sample_clients(self.num_clients)
            results = self.send_model_and_wait(targets=clients, data=model)
            model = self.aggregate(results, model, rnd)
            self.save_model(model)
        # evaluate the final global model on every client's val / test split
        self.send_model_and_wait(task_name="validate", targets=self.sample_clients(self.num_clients), data=model)
        self.info("Finished FedMoSA.")

    def aggregate(self, results, global_model, rnd):
        if not results:
            raise RuntimeError(f"Round {rnd}: no client results received")
        counts = [r.meta["mod_counts"] for r in results]
        sizes = [sum(c.values()) for c in counts]
        balanced = torch.tensor([n * len(c) for n, c in zip(sizes, counts)], dtype=torch.float64)
        balanced = balanced / balanced.sum()
        per_mod = {}
        for m in self.modalities:
            n = torch.tensor([c.get(m, 0) for c in counts], dtype=torch.float64)
            per_mod[m] = n / n.sum() if n.sum() > 0 else None

        params = {}
        for name, old in global_model.params.items():
            hit = EXPERT_RE.search(name)
            w = balanced if hit is None else per_mod[self.modalities[int(hit.group(1))]]
            if w is None:  # no client holds this modality: keep the global expert
                params[name] = old
                continue
            acc = None
            for wk, r in zip(w.tolist(), results):
                if wk == 0:
                    continue
                t = torch.as_tensor(r.params[name]).double() * wk
                acc = t if acc is None else acc + t
            # same container as the persistor's model (numpy), so clients decode it like the initial model
            params[name] = acc.to(torch.as_tensor(old).dtype).numpy()

        protos = None
        if all(r.meta.get("prototypes") for r in results):
            protos = []
            for b in range(len(results[0].meta["prototypes"])):
                protos.append(
                    {
                        k: sum(
                            torch.as_tensor(r.meta["prototypes"][b][k]).double() * wk
                            for wk, r in zip(balanced.tolist(), results)
                        ).tolist()
                        for k in results[0].meta["prototypes"][b]
                    }
                )
        mods = {m: sum(c.get(m, 0) for c in counts) for m in self.modalities}
        self.info(f"Round {rnd}: aggregated {len(results)} clients, samples per modality {mods}")
        return FLModel(params=params, meta={"prototypes": protos}, current_round=rnd)
