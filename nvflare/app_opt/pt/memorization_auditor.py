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

import random
import time
from abc import abstractmethod

import numpy as np
import torch

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.abstract.memorization_auditor import MemorizationAuditor, UnsupportedAudit
from nvflare.app_opt.pt.reference_scorer import tensor_digest


def membership_auc(members, nonmembers):
    """Directional AUROC with higher scores indicating membership; ties count half."""
    members, nonmembers = np.asarray(members, dtype=float), np.asarray(nonmembers, dtype=float)
    if members.ndim != 1 or nonmembers.ndim != 1 or not len(members) or not len(nonmembers):
        raise UnsupportedAudit("Both member and nonmember scores are required")
    if not np.isfinite(members).all() or not np.isfinite(nonmembers).all():
        raise ValueError("Nonfinite membership scores")
    sorted_nonmembers = np.sort(nonmembers)
    left = np.searchsorted(sorted_nonmembers, members, side="left")
    right = np.searchsorted(sorted_nonmembers, members, side="right")
    return float(np.mean((left + right) / (2 * len(nonmembers))))


class PTMemorizationAuditor(MemorizationAuditor):
    """Experimental PyTorch audit adapter with reconstruction and RNG isolation.

    Subclasses construct a separate module and evaluate fixed local records.
    evaluate() returns {"scores": {name: {"members": [...], "nonmembers": [...]}},
    "metrics": {...}}. Only aggregates enter the report. Record ordering must be
    identical for baseline and candidate scoring. Check the cooperative deadline
    between batches; this budget cannot preempt a blocked CUDA operation.
    """

    def __init__(self, device="cpu", max_seconds=120.0):
        super().__init__()
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        self.device = device
        self.max_seconds = max_seconds

    @abstractmethod
    def build_model(self):
        """Construct an independent module matching the complete parameter state."""

    @abstractmethod
    def evaluate(self, module, deadline: float) -> dict:
        """Score fixed local records without updating the model. Honor deadline."""

    @torch.no_grad()
    def audit(self, model, baseline, fl_ctx):
        if model.params_type not in (ParamsType.FULL, ParamsType.DIFF) or not isinstance(model.params, dict):
            raise UnsupportedAudit("Only full parameter dictionaries and matching differences are supported")
        if baseline is not None:
            if baseline.params_type != ParamsType.FULL or not isinstance(baseline.params, dict):
                raise UnsupportedAudit("Baseline must contain full parameters")
            if model.current_round != baseline.current_round:
                raise UnsupportedAudit("Baseline and candidate task rounds do not match")
        if model.params_type == ParamsType.DIFF:
            if baseline is None or baseline.params_type != ParamsType.FULL:
                raise UnsupportedAudit("Difference requires matching full baseline")
            if model.params.keys() != baseline.params.keys():
                raise UnsupportedAudit("Partial differences are unsupported")
            if any(np.shape(model.params[k]) != np.shape(baseline.params[k]) for k in model.params):
                raise UnsupportedAudit("Difference shapes do not match baseline")
            candidate = {k: torch.as_tensor(baseline.params[k]) + torch.as_tensor(v) for k, v in model.params.items()}
        else:
            candidate = {k: torch.as_tensor(v) for k, v in model.params.items()}
        deadline = time.monotonic() + self.max_seconds
        python_state, numpy_state = random.getstate(), np.random.get_state()
        device = torch.device(self.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                module = self.build_model().to(device).eval().requires_grad_(False)
                if module.state_dict().keys() != candidate.keys():
                    raise UnsupportedAudit("Partial or mismatched model state")
                module.load_state_dict(candidate, strict=True)
                result = self.evaluate(module, deadline)
                report = {
                    "model_fingerprint": tensor_digest(candidate),
                    "scorer": f"{type(self).__module__}.{type(self).__name__}",
                    "metrics": dict(result.get("metrics", {})),
                    "sample_counts": {},
                }
                for name, scores in result.get("scores", {}).items():
                    auc = membership_auc(scores["members"], scores["nonmembers"])
                    report["metrics"].update({f"{name}/auc": auc, f"{name}/nauc": 2 * abs(auc - 0.5)})
                    report["sample_counts"][name] = {k: len(scores[k]) for k in ("members", "nonmembers")}
                if not report["sample_counts"]:
                    raise UnsupportedAudit("No membership scores evaluated")
                if baseline is not None:
                    module.load_state_dict({k: torch.as_tensor(v) for k, v in baseline.params.items()}, strict=True)
                    before = self.evaluate(module, deadline)
                    report["baseline_fingerprint"] = tensor_digest(baseline.params)
                    for name, scores in result["scores"].items():
                        previous = before["scores"][name]
                        if any(len(scores[k]) != len(previous[k]) for k in ("members", "nonmembers")):
                            raise ValueError("Baseline and candidate audit records must align")
                        delta = {k: np.asarray(scores[k]) - np.asarray(previous[k]) for k in ("members", "nonmembers")}
                        auc = membership_auc(delta["members"], delta["nonmembers"])
                        report["metrics"].update({f"{name}/change_auc": auc, f"{name}/change_nauc": 2 * abs(auc - 0.5)})
                if time.monotonic() > deadline:
                    raise TimeoutError("Audit deadline exceeded")
                return report
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
