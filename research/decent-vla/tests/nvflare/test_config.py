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

"""fl.nvflare config-block parsing (no nvflare needed — pure loader)."""

from __future__ import annotations

import pytest
import yaml

from decent_vla.configs import load_run_config
from decent_vla.configs.loader import ConfigError


def _cfg(tmp_path, fl_extra):
    cfg = {
        "name": "t",
        "backend": {"name": "mock"},
        "data": {"partitioning": "iid", "num_clients": 2},
        "fl": {"rounds": 1, "local_steps": 1, "output_dir": str(tmp_path / "o"), **fl_extra},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_nvflare_block_defaults(tmp_path):
    cfg = load_run_config(_cfg(tmp_path, {"execution": "nvflare"}))
    assert cfg.fl.mode == "nvflare"
    nv = cfg.fl.nvflare
    assert nv.env == "simulator"
    assert nv.workspace is None
    assert nv.threads is None
    assert nv.gpu is None
    assert nv.site_prefix == ""
    assert nv.launch_external_process is False


def test_nvflare_block_fields(tmp_path):
    cfg = load_run_config(
        _cfg(
            tmp_path,
            {
                "execution": "nvflare",
                "nvflare": {"env": "export", "gpu": "0,1", "threads": 2,
                            "site_prefix": "site-", "launch_external_process": True},
            },
        )
    )
    nv = cfg.fl.nvflare
    assert nv.env == "export"
    assert nv.gpu == "0,1"
    assert nv.threads == 2
    assert nv.site_prefix == "site-"
    assert nv.launch_external_process is True


def test_nvflare_block_rejects_unknown_keys(tmp_path):
    with pytest.raises(ConfigError, match="fl.nvflare"):
        load_run_config(
            _cfg(tmp_path, {"execution": "nvflare", "nvflare": {"wokspace": "/tmp/x"}})
        )


def test_nvflare_block_absent_on_other_modes_is_default(tmp_path):
    cfg = load_run_config(_cfg(tmp_path, {"execution": "local"}))
    assert cfg.fl.nvflare.env == "simulator"
