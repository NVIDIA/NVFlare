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

"""Tiny synthetic components for functional integration, not privacy benchmarking."""

import hashlib
import time

import torch

from nvflare.app_opt.pt.memorization_auditor import PTMemorizationAuditor
from nvflare.app_opt.pt.reference_scorer import FrozenReferenceScorer
from nvflare.app_opt.pt.trap_loss import causal_token_log_probs


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 16)
        self.output = torch.nn.Linear(16, 16)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


def create_model(seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyLM()


def records(site, split):
    """Fixed, disjoint member/nonmember cohorts within each synthetic site."""
    seed = int.from_bytes(hashlib.sha256(site.encode()).digest()[:4], "big")
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, 16, (128, 8), generator=generator).unique(dim=0)
    if len(rows) < 96:
        raise RuntimeError("Insufficient distinct synthetic records")
    offset = {"members": 0, "nonmembers": 32, "reference": 64}[split]
    return rows[offset : offset + 32].clone()


def make_reference():
    model = create_model(seed=11)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    public = records("public-reference", "reference")
    for _ in range(5):
        optimizer.zero_grad()
        scores, mask = causal_token_log_probs(model(public), public)
        (-scores[mask].mean()).backward()
        optimizer.step()
    return FrozenReferenceScorer(model, "synthetic-vocab16-v1", "tiny-lm-length8-fp32-v1")


class SyntheticAuditor(PTMemorizationAuditor):
    def audit(self, model, baseline, fl_ctx):
        self.site = fl_ctx.get_identity_name()
        return super().audit(model, baseline, fl_ctx)

    def build_model(self):
        return TinyLM()

    def evaluate(self, module, deadline):
        cohorts = {}
        for split in ("members", "nonmembers"):
            if time.monotonic() > deadline:
                raise TimeoutError("Synthetic audit deadline exceeded")
            tokens = records(self.site, split).to(self.device)
            scores, mask = causal_token_log_probs(module(tokens), tokens)
            cohorts[split] = (scores.sum(dim=1) / mask.sum(dim=1)).cpu().tolist()
        return {"scores": {"likelihood": cohorts}}
