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

"""Pythia dossier-completion task and aggregate local auditor (synthetic data only)."""

import hashlib
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from nvflare.app_opt.pt.memorization_auditor import PTMemorizationAuditor
from nvflare.app_opt.pt.reference_scorer import FrozenReferenceScorer
from nvflare.app_opt.pt.trap_loss import causal_token_log_probs

MODEL_ID = "EleutherAI/pythia-70m-deduped"
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
MAX_LENGTH = 128
BATCH_SIZE = 2
LOCAL_STEPS = 8
ROUNDS = 2
SITES = ("site-1", "site-2")
SCORING_ID = "completion-only/shifted-labels/fp32/eager/max128/v1"


def configure(seed=17):
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("Set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 before starting this fp32 example")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(group, count):
    result = []
    for i in range(count):
        identity = f"{group}-{i:03d}"
        code = hashlib.sha256(identity.encode()).hexdigest()[:8]
        result.append({"id": identity, "prompt": f"Fictional dossier {identity}. Access code:", "answer": f" {code}"})
    return result


def make_cohorts():
    data = {"reference": records("public", 32), "validation": records("validation", 16)}
    for site in SITES:
        train = records(site, 16)
        data[site] = {"train": train, "members": train[:8], "nonmembers": records(f"{site}-heldout", 8)}
    data["final"] = {
        "members": [row for site in SITES for row in data[site]["train"][8:]],
        "nonmembers": records("final-heldout", 16),
    }
    return data


def batch(tokenizer, rows, device):
    """Concatenate separately tokenized prompt/completion; never truncate audit records."""
    ids, labels = [], []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(row["answer"], add_special_tokens=False)
        if not prompt or not answer or len(prompt) + len(answer) > MAX_LENGTH:
            raise ValueError("Empty or overlong record; change the cohort explicitly, not by silent truncation")
        ids.append(prompt + answer)
        labels.append([-100] * len(prompt) + answer)
    length = max(map(len, ids))
    attention = [[1] * len(row) + [0] * (length - len(row)) for row in ids]
    ids = [row + [tokenizer.eos_token_id] * (length - len(row)) for row in ids]
    labels = [row + [-100] * (length - len(row)) for row in labels]
    return {
        name: torch.tensor(value, device=device)
        for name, value in (("input_ids", ids), ("attention_mask", attention), ("labels", labels))
    }


def logits(module, tokens):
    return module(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], use_cache=False).logits


class CausalLM(torch.nn.Module):
    """Independent architecture; complete state includes the wrapped model's buffers."""

    def __init__(self, assets_dir):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(Path(assets_dir) / "base", local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", trust_remote_code=False)

    def forward(self, **kwargs):
        return self.model(**kwargs)


def load_state(module, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state, strict=True)
    return module


def tokenizer_for(assets_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(assets_dir) / "base", local_files_only=True, trust_remote_code=False)


def reference_for(assets_dir, device):
    assets = Path(assets_dir)
    manifest = json.loads((assets / "manifest.json").read_text())
    if sha256(assets / "reference.pt") != manifest["files"]["reference.pt"]:
        raise ValueError("Reference checkpoint changed")
    module = load_state(CausalLM(assets), assets / "reference.pt").to(device)
    return FrozenReferenceScorer(module, manifest["files"]["base/tokenizer.json"], SCORING_ID)


@contextmanager
def measurement(result, device):
    """Allocator peaks for this process/phase; host RSS is a process-lifetime high water mark."""
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize(device)
        result.update(
            elapsed_seconds=time.perf_counter() - started,
            host_rss_lifetime_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
        )


class DossierAuditor(PTMemorizationAuditor):
    def __init__(self, assets_dir, device="cuda:0", max_seconds=120.0, cohort=None):
        super().__init__(device=device, max_seconds=max_seconds)
        self.assets_dir = assets_dir
        self.cohort = cohort

    def build_model(self):
        return CausalLM(self.assets_dir)

    def audit(self, model, baseline, fl_ctx):
        self.active_cohort = self.cohort or fl_ctx.get_identity_name()
        resources = {}
        with measurement(resources, self.device):
            report = super().audit(model, baseline, fl_ctx)
        report["resources"] = resources
        return report

    def evaluate(self, module, deadline):
        data = json.loads((Path(self.assets_dir) / "cohorts.json").read_text())[self.active_cohort]
        tokenizer = tokenizer_for(self.assets_dir)
        reference = reference_for(self.assets_dir, self.device)
        scores = {name: {split: [] for split in ("members", "nonmembers")} for name in ("likelihood", "tra")}
        total_loss, correct, count = 0.0, 0, 0
        for split in ("members", "nonmembers"):
            for offset in range(0, len(data[split]), BATCH_SIZE):
                if time.monotonic() > deadline:
                    raise TimeoutError("Dossier audit budget exceeded between batches")
                tokens = batch(tokenizer, data[split][offset : offset + BATCH_SIZE], self.device)
                output = logits(module, tokens)
                target, mask = causal_token_log_probs(output, tokens["labels"])
                ref, ref_mask = reference.score(
                    tokens["input_ids"], tokens["labels"], attention_mask=tokens["attention_mask"], use_cache=False
                )
                if not torch.equal(mask, ref_mask):
                    raise ValueError("Reference mask changed")
                scores["likelihood"][split].extend((target.sum(1) / mask.sum(1)).cpu().tolist())
                scores["tra"][split].extend(((target - ref).sum(1) / mask.sum(1)).cpu().tolist())
                total_loss -= target[mask].sum().item()
                correct += ((output[:, :-1].argmax(-1) == tokens["labels"][:, 1:]) & mask).sum().item()
                count += mask.sum().item()
        return {
            "scores": scores,
            "metrics": {
                "completion_ce": total_loss / count,
                "completion_token_accuracy": correct / count,
                "completion_tokens": count,
            },
        }
