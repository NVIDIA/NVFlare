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

"""No config field may be silently dropped on the YAML → dataclass hop.

`EvalConfig.n_action_steps` was added to the schema, honored by
`apply_action_horizon`, recorded in the eval cache provenance key, and set in
thirteen run configs — while `loader.py::_build_eval` never read it. Every unit
test passed. The feature was inert for its entire life, and months of LIBERO
evals ran fully open loop (`n_action_steps=50`) inside directories named
`_nact10`, which is worth ~33 points on an identical checkpoint.

The builders that splat `**d` cannot have this bug. The ones that enumerate
kwargs by hand can, and get one more field each time a schema grows. So the
check is behavioural: hand each builder a dict with a non-default value for
every scalar field it owns, and assert none of them come back at the default.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from decent_vla.configs import loader, schemas

# Builders that enumerate their kwargs by hand — the ones that can drop a field.
EXPLICIT_BUILDERS = [
    ("_build_eval", schemas.EvalConfig),
    ("_build_fl", schemas.FLConfig),
    ("_build_benchmark", schemas.BenchmarkConfig),
]

# Fields whose value is a nested dataclass built by another builder, or whose
# meaning is positional rather than scalar — covered by their own round trips.
NESTED = {"benchmark", "strategy", "nvflare", "optim", "lora", "model", "extra"}


# Fields validated against a fixed vocabulary: a made-up string is *correctly*
# rejected, so probe with a different LEGAL value instead of a nonsense one.
ALTERNATIVES = {"wire_dtype": "fp32", "mode": "nvflare"}


def _sentinel(field: dc.Field):
    """A legal value that cannot be confused with the field's default."""
    if field.name in ALTERNATIVES:
        return ALTERNATIVES[field.name]
    default = field.default if field.default is not dc.MISSING else None
    if default is dc.MISSING or field.default_factory is not dc.MISSING:  # type: ignore[misc]
        default = (field.default_factory() if field.default_factory is not dc.MISSING  # type: ignore[misc]
                   else None)
    # list BEFORE str: "list[str]" contains "str", and handing a bare string to a
    # list field makes list("probe") == ['p','r','o','b','e'] and fakes a failure.
    if isinstance(default, list) or "list" in str(field.type):
        return ["probe-a", "probe-b"]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int) and not isinstance(default, bool):
        return default + 4242
    if isinstance(default, float):
        return default + 1.5
    if isinstance(default, str):
        return default + "-probe"
    ann = str(field.type)
    if "bool" in ann:
        return True
    if "int" in ann:
        return 4242
    if "float" in ann:
        return 1.5
    if "str" in ann:
        return "probe"
    return None


@pytest.mark.parametrize("builder_name,klass", EXPLICIT_BUILDERS,
                         ids=[b for b, _ in EXPLICIT_BUILDERS])
def test_builder_forwards_every_scalar_field(builder_name, klass):
    builder = getattr(loader, builder_name)
    payload, expected = {}, {}
    for f in dc.fields(klass):
        if f.name in NESTED:
            continue
        v = _sentinel(f)
        if v is None:            # nothing meaningful to probe with
            continue
        payload[f.name] = v
        expected[f.name] = v

    built = builder(payload)
    dropped = [n for n, v in expected.items() if getattr(built, n) != v]
    assert not dropped, (
        f"{builder_name} ignores {dropped} — these are set in YAML, stored on "
        f"{klass.__name__}, and never arrive. Add them to the constructor call."
    )
