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

import hashlib
import json
import os
import tempfile
import weakref
from pathlib import Path

import torch

from nvflare.app_opt.pt.trap_loss import causal_token_log_probs
from nvflare.fuel.utils.experimental import experimental


def tensor_digest(tensors):
    """Content fingerprint for named dense tensors, independent of device."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenReferenceScorer:
    """Experimental frozen causal-LM scorer with optional local tensor cache.

    Keep this object separate from the target module and its optimizer. Supply a
    tokenizer fingerprint and context/scoring identity covering tokenization and
    model configuration. The reference must remain immutable after construction;
    ordinary in-place changes, dtype/device conversions and storage replacement
    are rejected. In-place writes through .data or external storage aliases bypass
    version tracking and are unsupported. Recreate the scorer after any change.
    Cached scores are potentially sensitive local artifacts.
    """

    @experimental("FrozenReferenceScorer API may change")
    def __init__(self, model, tokenizer_fingerprint: str, scoring_identity: str, cache_dir=None):
        if not tokenizer_fingerprint or not scoring_identity:
            raise ValueError("Tokenizer and scoring identities are required")
        self.model = model.eval().requires_grad_(False)
        self.reference_fingerprint = tensor_digest(model.state_dict())
        self.identity = json.dumps(
            ["trap-reference-v1", self.reference_fingerprint, tokenizer_fingerprint, scoring_identity], sort_keys=True
        )
        self._versions = self._model_versions()
        # IDs can be recycled after a tensor is replaced. Weak references detect
        # that replacement without retaining old CPU/GPU tensor allocations.
        self._tensor_refs = [weakref.ref(t) for t in list(model.parameters()) + list(model.buffers())]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_hits = 0
        self.cache_misses = 0

    def _model_versions(self):
        return [
            (name, id(t), t._version, t.device, t.dtype, tuple(t.shape), t.stride(), t.data_ptr())
            for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers())
        ]

    @torch.no_grad()
    def score(self, input_ids, labels, **model_inputs):
        if any(ref() is None for ref in self._tensor_refs) or self._versions != self._model_versions():
            raise ValueError("Reference changed; construct a new FrozenReferenceScorer")
        self.model.eval()
        inputs = {"input_ids": input_ids, **model_inputs}
        path = None
        if self.cache_dir is not None:
            key = hashlib.sha256((self.identity + tensor_digest({**inputs, "labels": labels})).encode()).hexdigest()
            path = self.cache_dir / f"{key}.pt"
        if path and path.exists():
            cached = torch.load(path, map_location=input_ids.device, weights_only=True)
            self.cache_hits += 1
            return cached["log_probs"], cached["mask"]
        output = self.model(**inputs)
        logits = output.logits if hasattr(output, "logits") else output
        log_probs, mask = causal_token_log_probs(logits, labels)
        self.cache_misses += 1
        if path:
            fd, temporary = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save({"log_probs": log_probs.cpu(), "mask": mask.cpu()}, stream)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return log_probs, mask
