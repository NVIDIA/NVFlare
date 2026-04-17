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

import json
from unittest.mock import patch

import pytest


def test_dynamic_log_config_accepts_inline_json_string(tmp_path):
    from nvflare.fuel.utils.log_utils import dynamic_log_config

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"console": {"class": "logging.StreamHandler", "level": "INFO"}},
        "root": {"handlers": ["console"], "level": "INFO"},
    }

    with patch("nvflare.fuel.utils.log_utils.apply_log_config") as mock_apply:
        dynamic_log_config(json.dumps(config), str(tmp_path), str(tmp_path / "reload.json"))

    mock_apply.assert_called_once()
    assert mock_apply.call_args.args[0] == config


def test_dynamic_log_config_invalid_inline_json_raises_value_error(tmp_path):
    from nvflare.fuel.utils.log_utils import dynamic_log_config

    with pytest.raises(ValueError, match="Invalid dictConfig JSON"):
        dynamic_log_config('{"version": 1,', str(tmp_path), str(tmp_path / "reload.json"))


def test_log_modes_preserve_concise_and_add_msg_only():
    from nvflare.fuel.utils.log_utils import LogMode, logmode_config_dict

    assert logmode_config_dict[LogMode.CONCISE]["formatters"]["consoleFormatter"]["fmt"] == (
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    assert logmode_config_dict[LogMode.MSG_ONLY]["formatters"]["consoleFormatter"]["fmt"] == "%(message)s"


def test_validate_site_log_config_accepts_levels_and_modes():
    from nvflare.fuel.utils.log_utils import LogMode, validate_site_log_config

    assert validate_site_log_config("INFO") == "INFO"
    assert validate_site_log_config("20") == "20"
    assert validate_site_log_config(LogMode.MSG_ONLY) == LogMode.MSG_ONLY


def test_validate_site_log_config_rejects_dicts_and_file_paths():
    from nvflare.fuel.utils.log_utils import validate_site_log_config

    with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
        validate_site_log_config({"version": 1})

    with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
        validate_site_log_config("/my workspace/log.conf")
