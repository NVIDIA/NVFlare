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

import math
from numbers import Real
from typing import Optional

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants


def normalize_fedprox_mu(fedprox_mu: Optional[float]) -> Optional[float]:
    """Validate and normalize the optional FedProx coefficient."""
    if fedprox_mu is None:
        return None
    if isinstance(fedprox_mu, bool) or not isinstance(fedprox_mu, Real):
        raise TypeError("fedprox_mu must be a finite non-negative number or None.")

    fedprox_mu = float(fedprox_mu)
    if not math.isfinite(fedprox_mu) or fedprox_mu < 0.0:
        raise ValueError("fedprox_mu must be a finite non-negative number or None.")
    return None if fedprox_mu == 0.0 else fedprox_mu


def validate_fedprox_mu(fedprox_mu: float) -> float:
    """Validate the strictly positive FedProx coefficient required by a FedProx recipe or client."""
    if isinstance(fedprox_mu, bool) or not isinstance(fedprox_mu, Real):
        raise TypeError("fedprox_mu must be a finite positive number.")

    fedprox_mu = float(fedprox_mu)
    if not math.isfinite(fedprox_mu) or fedprox_mu <= 0.0:
        raise ValueError("fedprox_mu must be a finite positive number.")
    return fedprox_mu


def get_fedprox_mu(model: FLModel) -> float:
    """Read and validate the required positive FedProx coefficient from model metadata."""
    meta = model.meta or {}
    if AlgorithmConstants.FEDPROX_MU not in meta:
        raise ValueError(
            f"FedProx client requires positive {AlgorithmConstants.FEDPROX_MU!r} metadata on every training round."
        )
    try:
        return validate_fedprox_mu(meta[AlgorithmConstants.FEDPROX_MU])
    except (TypeError, ValueError) as e:
        raise ValueError(f"FedProx client received invalid {AlgorithmConstants.FEDPROX_MU!r} metadata: {e}") from e


def set_fedprox_metadata(model: FLModel, fedprox_mu: Optional[float]) -> None:
    """Set the reserved FedProx contract metadata without retaining stale values."""
    meta = dict(model.meta or {})
    if fedprox_mu is None:
        meta.pop(AlgorithmConstants.FEDPROX_MU, None)
    else:
        meta[AlgorithmConstants.FEDPROX_MU] = fedprox_mu
    model.meta = meta
