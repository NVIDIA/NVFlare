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

"""runtime/wire.py — the ONE wire-precision table both execution planes share.

These tests exist because `fl.wire_dtype` used to be applied asymmetrically:
`update_to_flmodel`/`update_to_recorddict` cast the client->server delta, while
`round_flmodel`/`round_message_content` shipped the fp32 master untouched. The
knob therefore meant half of what its name said, and at pi0.5's 3.62B params
the unconverted downlink was the single largest host-RAM term in the system
(13.5 GB per live copy vs 6.7 GB for the delta). The contract is now: one knob,
both directions.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from decent_vla.runtime import wire


def test_scalar_applies_to_both_directions():
    assert wire.resolve("fp16") == (torch.float16, torch.float16)
    assert wire.resolve("fp32") == (torch.float32, torch.float32)


def test_mapping_sets_each_direction():
    assert wire.resolve({"broadcast": "fp32", "delta": "fp16"}) == (
        torch.float32,
        torch.float16,
    )
    # An omitted direction falls back to the documented default, not to fp32.
    assert wire.resolve({"delta": "fp32"}) == (torch.float16, torch.float32)


def test_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="wire_dtype"):
        wire.resolve("bf16")


def test_rejects_misspelled_direction():
    """A typo must fail loudly: silently ignoring it would leave the run on the
    default precision while the operator believed otherwise."""
    with pytest.raises(ValueError, match="direction"):
        wire.resolve({"broadcst": "fp32"})


def test_bf16_is_always_downcast_to_fp16_never_widened():
    """numpy cannot carry bf16. Widening it to fp32
    would double the payload for zero information gain, so bf16 pins to fp16
    even when the configured dtype is fp32."""
    t = torch.randn(4, dtype=torch.bfloat16)
    assert wire.cast(t, torch.float32).dtype == torch.float16
    assert wire.cast(t, torch.float16).dtype == torch.float16


def test_integer_tensors_are_never_cast():
    assert wire.cast(torch.arange(4), torch.float16).dtype == torch.int64


def test_names_and_nbytes_agree_per_direction():
    spec = {"broadcast": "fp32", "delta": "fp16"}
    assert wire.names(spec) == ("fp32", "fp16")
    assert wire.nbytes(spec) == (4, 2)


def test_fp16_roundtrip_error_stays_within_one_ulp():
    """The precision claim behind defaulting to fp16: at most 2**-11 relative.
    Measured on the real pi05_base checkpoint the worst case was 4.88e-4, i.e.
    exactly this bound, with zero overflow."""
    t = torch.randn(4096) * 10.0
    rt = wire.cast(t, torch.float16).float()
    rel = (rt - t).abs() / t.abs().clamp_min(1e-3)
    assert rel.max() <= 2 ** -10


def test_codec_resolves_through_this_table():
    """One config has to mean the same bytes on the wire: the nvflare codec
    resolves its precision through this table. This test pins that property."""
    from decent_vla.runtime.nvflare import codec as ncodec

    try:
        for spec in ("fp16", "fp32", {"broadcast": "fp32", "delta": "fp16"}):
            ncodec.set_wire_dtype(spec)
            assert (ncodec._BROADCAST_DTYPE, ncodec._DELTA_DTYPE) == wire.resolve(spec)
    finally:
        ncodec.set_wire_dtype("fp16")


def _write_cfg(tmp_path, fl_extra):
    cfg = {
        "name": "t",
        "backend": {"name": "mock"},
        "data": {"partitioning": "task", "num_clients": 2, "batch_size": 4},
        "fl": {"strategy": {"name": "fedavg"}, "rounds": 1, "local_steps": 1, **fl_extra},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_config_defaults_are_fp16_and_lean(tmp_path):
    from decent_vla.configs import load_run_config

    cfg = load_run_config(str(_write_cfg(tmp_path, {})))
    assert cfg.fl.wire_dtype == "fp16"
    assert cfg.fl.lean_host_memory is True


def test_config_accepts_mapping_and_lean_opt_out(tmp_path):
    from decent_vla.configs import load_run_config

    cfg = load_run_config(
        str(_write_cfg(tmp_path, {
            "wire_dtype": {"broadcast": "fp32", "delta": "fp16"},
            "lean_host_memory": False,
        }))
    )
    assert wire.resolve(cfg.fl.wire_dtype) == (torch.float32, torch.float16)
    assert cfg.fl.lean_host_memory is False


def test_config_rejects_bad_wire_dtype_at_load_time(tmp_path):
    """Fail fast: a typo must not surface only at the first broadcast, minutes
    into a run that has already loaded a 14 GB model onto the GPUs."""
    from decent_vla.configs.loader import ConfigError, load_run_config

    with pytest.raises(ConfigError, match="wire_dtype"):
        load_run_config(str(_write_cfg(tmp_path, {"wire_dtype": "float8"})))
