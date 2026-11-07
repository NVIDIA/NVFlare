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

"""Bounded CUDA parity/resource validation of the actual SDK, not a new efficacy benchmark."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hf_model import (
    BATCH_SIZE,
    CausalLM,
    DossierAuditor,
    batch,
    configure,
    load_state,
    logits,
    measurement,
    reference_for,
    tokenizer_for,
    write_json,
)

from nvflare.apis.fl_context import FLContextManager
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.filters.memorization_audit_filter import AuditModelSnapshot
from nvflare.app_opt.pt.trap_loss import PTTRAPLoss, causal_token_log_probs


def independent_tokens(output, labels):
    """Negative cross entropy, independently shifted/masked (not the SDK helper)."""
    observed = labels[:, 1:]
    loss = F.cross_entropy(
        output[:, :-1].float().reshape(-1, output.shape[-1]), observed.reshape(-1), ignore_index=-100, reduction="none"
    )
    return -loss.reshape(observed.shape), observed != -100


def pair_auc(members, nonmembers):
    return sum(float(m > n) + 0.5 * float(m == n) for m in members for n in nonmembers) / (
        len(members) * len(nonmembers)
    )


@torch.no_grad()
def independent_audit(module, reference, tokenizer, cohorts, device):
    scores = {name: {split: [] for split in ("members", "nonmembers")} for name in ("likelihood", "tra")}
    loss, correct, count = 0.0, 0, 0
    for split in ("members", "nonmembers"):
        for offset in range(0, len(cohorts[split]), BATCH_SIZE):
            tokens = batch(tokenizer, cohorts[split][offset : offset + BATCH_SIZE], device)
            output = logits(module, tokens)
            target, mask = independent_tokens(output, tokens["labels"])
            ref, _ = independent_tokens(logits(reference, tokens), tokens["labels"])
            scores["likelihood"][split].extend((target.sum(1) / mask.sum(1)).cpu().tolist())
            scores["tra"][split].extend(((target - ref).sum(1) / mask.sum(1)).cpu().tolist())
            loss -= target[mask].sum().item()
            correct += ((output[:, :-1].argmax(-1) == tokens["labels"][:, 1:]) & mask).sum().item()
            count += mask.sum().item()
    return scores, {
        "completion_ce": loss / count,
        "completion_token_accuracy": correct / count,
        "completion_tokens": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("This representative validation requires CUDA")
    device = "cuda:0"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    assets = Path(args.assets)
    data = json.loads((assets / "cohorts.json").read_text())
    tokenizer = tokenizer_for(assets)
    resources = {}
    with measurement(resources.setdefault("reference_setup", {}), device):
        reference = reference_for(assets, device)
    module = load_state(CausalLM(assets), assets / "initial.pt").to(device).eval()
    baseline = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    tokens = batch(tokenizer, data["site-1"]["train"][:BATCH_SIZE], device)
    with measurement(resources.setdefault("reference_scoring", {}), device):
        ref, mask = reference.score(
            tokens["input_ids"], tokens["labels"], attention_mask=tokens["attention_mask"], use_cache=False
        )
    with torch.no_grad():
        ref_manual, manual_mask = independent_tokens(logits(reference.model, tokens), tokens["labels"])
    torch.testing.assert_close(ref, ref_manual, atol=1e-5, rtol=1e-6)
    assert torch.equal(mask, manual_mask)
    trap = PTTRAPLoss(weight=0.5, threshold=0.0)
    with measurement(resources.setdefault("sdk_loss_backward", {}), device):
        module.zero_grad(set_to_none=True)
        target, mask = causal_token_log_probs(logits(module, tokens), tokens["labels"])
        active = ((target > ref) & mask).sum().item()
        assert active > 0, "Parity must exercise a nonzero TRAP penalty"
        sdk_penalty = trap(target, ref, mask)
        sdk_loss = -target[mask].mean() + sdk_penalty
        sdk_loss.backward()
    gradients = {k: p.grad.detach().cpu().clone() for k, p in module.named_parameters() if p.grad is not None}
    with measurement(resources.setdefault("independent_loss_backward", {}), device):
        module.zero_grad(set_to_none=True)
        manual, manual_mask = independent_tokens(logits(module, tokens), tokens["labels"])
        manual_penalty = (
            0.5
            * torch.maximum(manual[manual_mask] - ref_manual[manual_mask], torch.zeros_like(manual[manual_mask])).mean()
        )
        manual_loss = -manual[manual_mask].mean() + manual_penalty
        manual_loss.backward()
    torch.testing.assert_close(sdk_loss, manual_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sdk_penalty, manual_penalty, atol=1e-6, rtol=1e-6)
    gradient_error = 0.0
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            actual = parameter.grad.detach().cpu()
            torch.testing.assert_close(gradients[name], actual, atol=1e-6, rtol=1e-4)
            gradient_error = max(gradient_error, (gradients[name] - actual).abs().max().item())
    # A single bounded update creates a nontrivial candidate for the audit comparisons.
    torch.optim.SGD(module.parameters(), lr=0.001).step()
    module.zero_grad(set_to_none=True)
    del gradients
    candidate = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    context = FLContextManager(identity_name="site-1", job_id="sdk-parity").new_context()
    baseline_model = FLModel(params=baseline, current_round=0)
    snapshot = AuditModelSnapshot()
    with measurement(resources.setdefault("baseline_snapshot", {}), device):
        snapshot.capture(baseline_model, context)
        copied = snapshot.pop(baseline_model, context)
    assert copied is not None
    assert all(copied.params[k].data_ptr() != v.data_ptr() for k, v in baseline.items())
    del copied
    auditor = DossierAuditor(str(assets))
    with measurement(resources.setdefault("full_audit", {}), device):
        report = auditor.audit(FLModel(params=candidate, current_round=0), baseline_model, context)
    candidate_scores, metrics = independent_audit(module, reference.model, tokenizer, data["site-1"], device)
    module.load_state_dict(baseline)
    baseline_scores, _ = independent_audit(module, reference.model, tokenizer, data["site-1"], device)
    for name, values in candidate_scores.items():
        auc = pair_auc(values["members"], values["nonmembers"])
        delta = {k: np.array(values[k]) - np.array(baseline_scores[name][k]) for k in values}
        change = pair_auc(delta["members"], delta["nonmembers"])
        metrics.update(
            {
                f"{name}/auc": auc,
                f"{name}/nauc": 2 * abs(auc - 0.5),
                f"{name}/change_auc": change,
                f"{name}/change_nauc": 2 * abs(change - 0.5),
            }
        )
    for name, value in metrics.items():
        assert abs(report["metrics"][name] - value) <= 1e-6, name
    differences = {k: candidate[k] - baseline[k] for k in candidate}
    reconstructed = {k: baseline[k] + differences[k] for k in baseline}
    for key in baseline:
        torch.testing.assert_close(reconstructed[key], candidate[key], atol=1e-7, rtol=1e-6)
    with measurement(resources.setdefault("difference_audit", {}), device):
        diff_report = auditor.audit(
            FLModel(params=differences, params_type=ParamsType.DIFF, current_round=0), baseline_model, context
        )
    reconstructed_report = auditor.audit(FLModel(params=reconstructed, current_round=0), baseline_model, context)
    assert diff_report["metrics"] == reconstructed_report["metrics"]
    assert diff_report["model_fingerprint"] == reconstructed_report["model_fingerprint"]
    # Cooperative timeout is checked after construction/between batches, not a hard interrupt.
    timeout_started = time.monotonic()
    try:
        DossierAuditor(str(assets), max_seconds=1e-6).audit(FLModel(params=candidate, current_round=0), None, context)
    except TimeoutError:
        timeout_observed_seconds = time.monotonic() - timeout_started
    else:
        raise AssertionError("Expected cooperative timeout")
    write_json(
        output / "parity.json",
        {
            "passed": True,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "tolerances": {
                "reference_atol": 1e-5,
                "reference_rtol": 1e-6,
                "loss_atol": 1e-6,
                "loss_rtol": 1e-6,
                "gradient_atol": 1e-6,
                "gradient_rtol": 1e-4,
                "audit_atol": 1e-6,
            },
            "reference_max_abs_error": (ref - ref_manual).abs().max().item(),
            "loss_abs_error": abs(sdk_loss.item() - manual_loss.item()),
            "gradient_max_abs_error": gradient_error,
            "gradient_tensors": sum(p.requires_grad for p in module.parameters()),
            "active_trap_tokens": active,
            "valid_tokens": mask.sum().item(),
            "input_shape": list(tokens["input_ids"].shape),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "trap_penalty": sdk_penalty.item(),
            "full_state_bytes": sum(v.numel() * v.element_size() for v in baseline.values()),
            "audit_metrics": report["metrics"],
            "audit_sample_counts": report["sample_counts"],
            "difference_parity": True,
            "cooperative_timeout_observed_seconds": timeout_observed_seconds,
            "resources": resources,
        },
    )


if __name__ == "__main__":
    main()
