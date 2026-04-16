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
