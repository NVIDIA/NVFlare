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

"""Two-round full-state Client API training; the local reference never participates in aggregation."""

import argparse
import json
from pathlib import Path

import torch
from hf_model import (
    BATCH_SIZE,
    LOCAL_STEPS,
    CausalLM,
    batch,
    configure,
    logits,
    measurement,
    reference_for,
    tokenizer_for,
    write_json,
)

import nvflare.client as flare
from nvflare.app_opt.pt.trap_loss import PTTRAPLoss, causal_token_log_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure()
    flare.init()
    site = flare.get_site_name()
    data = json.loads((Path(args.assets) / "cohorts.json").read_text())
    tokenizer = tokenizer_for(args.assets)
    module = CausalLM(args.assets).to(args.device)
    reference = reference_for(args.assets, args.device)
    trap = PTTRAPLoss(weight=0.5, threshold=0.0)
    while flare.is_running():
        incoming = flare.receive()
        if incoming is None:
            break
        module.load_state_dict(incoming.params, strict=True)
        module.eval()
        with torch.no_grad():
            tokens = batch(tokenizer, data["validation"], args.device)
            scores, mask = causal_token_log_probs(logits(module, tokens), tokens["labels"])
            validation_ce = -scores[mask].mean().item()
        resources = {}
        with measurement(resources, args.device):
            module.train()
            optimizer = torch.optim.AdamW(module.parameters(), lr=5e-5)
            for step in range(LOCAL_STEPS):
                rows = data[site]["train"][step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
                tokens = batch(tokenizer, rows, args.device)
                ref, ref_mask = reference.score(
                    tokens["input_ids"], tokens["labels"], attention_mask=tokens["attention_mask"], use_cache=False
                )
                optimizer.zero_grad(set_to_none=True)
                target, mask = causal_token_log_probs(logits(module, tokens), tokens["labels"])
                if not torch.equal(mask, ref_mask):
                    raise ValueError("Target and reference masking differ")
                loss = -target[mask].mean() + trap(target, ref, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()
            state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        write_json(Path(args.measurements) / f"{site}-round-{incoming.current_round}.json", resources)
        flare.send(
            flare.FLModel(
                params=state,
                metrics={"ce": validation_ce},
                current_round=incoming.current_round,
                meta={"NUM_STEPS_CURRENT_ROUND": LOCAL_STEPS},
            )
        )


if __name__ == "__main__":
    main()
