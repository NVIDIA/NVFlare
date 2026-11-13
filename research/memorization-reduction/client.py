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

"""CPU Client API loop with a local frozen reference and additive TRAP loss."""

import torch
from model import create_model, make_reference, records

import nvflare.client as flare
from nvflare.app_opt.pt.trap_loss import PTTRAPLoss, causal_token_log_probs


def main():
    torch.set_num_threads(1)
    flare.init()
    site = flare.get_site_name()
    model = create_model()
    reference = make_reference()
    members, nonmembers = records(site, "members"), records(site, "nonmembers")
    public = records("public-reference", "reference")
    if {tuple(r) for r in public.tolist()} & {tuple(r) for r in members.tolist()}:
        raise RuntimeError("Synthetic reference overlaps protected records")
    reference_scores, reference_mask = reference.score(members, members)
    trap = PTTRAPLoss(weight=0.5, threshold=0.0)
    while flare.is_running():
        incoming = flare.receive()
        if incoming is None:
            break
        model.load_state_dict(incoming.params)
        model.eval()
        with torch.no_grad():
            scores, mask = causal_token_log_probs(model(nonmembers), nonmembers)
            before_ce = -scores[mask].mean().item()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            scores, mask = causal_token_log_probs(model(members), members)
            if not torch.equal(mask, reference_mask):
                raise ValueError("Reference and target tokens must align")
            loss = -scores[mask].mean() + trap(scores, reference_scores, mask)
            loss.backward()
            optimizer.step()
        flare.send(
            flare.FLModel(
                params={name: value.detach().clone() for name, value in model.state_dict().items()},
                metrics={"ce": before_ce},
                current_round=incoming.current_round,
                meta={"NUM_STEPS_CURRENT_ROUND": 3},
            )
        )


if __name__ == "__main__":
    main()
