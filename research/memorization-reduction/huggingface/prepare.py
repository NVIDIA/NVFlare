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

"""Prepare disjoint synthetic cohorts and a public fine-tuned frozen reference."""

import argparse
from pathlib import Path

import torch
from hf_model import (
    MODEL_ID,
    REVISION,
    CausalLM,
    batch,
    configure,
    logits,
    make_cohorts,
    measurement,
    sha256,
    write_json,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from nvflare.app_opt.pt.trap_loss import causal_token_log_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure()
    assets = Path(args.assets).resolve()
    assets.mkdir(parents=True, exist_ok=False)
    data = make_cohorts()
    write_json(assets / "cohorts.json", data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.float32, attn_implementation="eager", trust_remote_code=False
    )
    pretrained.save_pretrained(assets / "base")
    tokenizer.save_pretrained(assets / "base")
    module = CausalLM(assets)
    module.model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    torch.save(module.state_dict(), assets / "initial.pt")
    resources = {}
    with measurement(resources, args.device):
        module.to(args.device).train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.001)
        for step in range(16):
            tokens = batch(tokenizer, data["reference"][2 * step : 2 * step + 2], args.device)
            optimizer.zero_grad(set_to_none=True)
            scores, mask = causal_token_log_probs(logits(module, tokens), tokens["labels"])
            (-scores[mask].mean()).backward()
            optimizer.step()
        torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, assets / "reference.pt")
    files = {str(p.relative_to(assets)): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file()}
    write_json(
        assets / "manifest.json",
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "reference_steps": 16,
            "reference_resources": resources,
            "files": files,
        },
    )


if __name__ == "__main__":
    main()
