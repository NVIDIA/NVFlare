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

from collections import Counter

import pytest

from nvflare.app_common.ccwf.eval_gen import parallel_eval_generator

SITES = ["site-1", "site-2", "site-3", "site-4"]


def _site_loads(evals):
    # an evaluator fetches the model from its owner, so both sites of an
    # evaluation are busy with it; a self-evaluation is one action
    loads = Counter()
    for evaluator, evaluatee in evals:
        loads[evaluator] += 1
        if evaluatee != evaluator:
            loads[evaluatee] += 1
    return loads


@pytest.mark.parametrize("max_parallel_actions", [1, 2, 3])
@pytest.mark.parametrize("evaluatees", [SITES, SITES[:1]], ids=["all sites", "one model owner"])
def test_no_site_takes_more_actions_than_allowed(max_parallel_actions, evaluatees):
    for evals in parallel_eval_generator(list(SITES), list(evaluatees), max_parallel_actions):
        assert max(_site_loads(evals).values()) <= max_parallel_actions, evals


@pytest.mark.parametrize("max_parallel_actions", [1, 2, 3])
def test_every_evaluation_is_generated_once(max_parallel_actions):
    generated = [e for evals in parallel_eval_generator(list(SITES), list(SITES), max_parallel_actions) for e in evals]

    assert sorted(generated) == sorted((t, e) for t in SITES for e in SITES)


def test_zero_parallel_actions_is_rejected():
    # no evaluation could ever be scheduled, and the generator would never finish
    with pytest.raises(ValueError):
        next(parallel_eval_generator(list(SITES), list(SITES), 0))
