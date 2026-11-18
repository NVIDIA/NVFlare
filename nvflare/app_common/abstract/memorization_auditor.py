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

from abc import ABC, abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.fuel.utils.experimental import experimental


class UnsupportedAudit(ValueError):
    """The auditor cannot interpret this model or observation."""


class MemorizationAuditor(FLComponent, ABC):
    """Experimental site-local audit contract, also usable for final evaluation.

    Implementations must leave inputs and training RNG state unchanged, enforce
    their sample/runtime budgets, and return JSON-compatible aggregate results.
    Raise UnsupportedAudit for unsupported representations, not a zero score.
    Raw examples, prompts, and record scores must not be included in summaries.
    """

    @experimental("MemorizationAuditor API may change")
    def __init__(self):
        super().__init__()

    @abstractmethod
    def audit(self, model: FLModel, baseline: FLModel | None, fl_ctx: FLContext) -> dict:
        """Return metrics, sample counts, artifact fingerprints, and scorer identity."""
