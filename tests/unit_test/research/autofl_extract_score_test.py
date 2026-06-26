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

import importlib.util
import sys
from pathlib import Path


def _load_extract_score():
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "research" / "auto-fl-research" / "scripts" / "extract_score.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_extract_score", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_score_prefers_explicit_test_accuracy():
    extract_score = _load_extract_score()
    data = {
        "site-1": {
            extract_score.PRIMARY_MODEL_KEY: {
                "accuracy": 0.5,
                "test_accuracy": 0.8,
            }
        }
    }

    assert extract_score.extract_score(data) == 0.8
