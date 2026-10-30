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

"""Optional XGBoost worker pools must retain incomplete CPU attribution."""

from concurrent import futures
from decimal import Decimal

import pytest

from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant

pytest.importorskip("xgboost")

from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he.adder import Adder
from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he.decrypter import Decrypter
from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he.encryptor import Encryptor


@pytest.mark.parametrize("pool_kind", ["encryptor", "adder", "decrypter"])
def test_xgboost_process_pools_do_not_claim_complete_cpu(monkeypatch, pool_kind):
    values = iter(((Decimal(0), Decimal(0)), (Decimal(1), Decimal(0))))
    accountant = CpuConsumedAccountant(usage_reader=lambda: next(values))
    monkeypatch.setattr(cpu_module, "_active", accountant)
    monkeypatch.setattr(futures, "ProcessPoolExecutor", lambda **_kwargs: object())
    if pool_kind == "encryptor":
        Encryptor(pubkey="test")
    elif pool_kind == "adder":
        Adder()
    else:
        Decrypter(private_key="test")

    assert accountant.finish() == {"status": "partial", "issues": ["attribution_incomplete"], "cpu_seconds": "1"}
