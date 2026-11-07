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

"""Paths and configuration for this example only."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
PIN = "3310a1f61b345951adc29cdb8f12f9e1d982f88c"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_environment():
    for line in (ROOT / ".env").read_text().splitlines():
        name, value = line.split("=", 1)
        os.environ[name] = value
    os.environ["UNITY_CATALOG_URL"] = "http://127.0.0.1:18080/api/2.1/unity-catalog"
    os.environ["CATALOG_DATABASE_URL"] = (
        "postgresql+psycopg://catalog_reader:local-demo-reader@127.0.0.1:55432/sanctuary"
    )


def prod_dir():
    from nvflare.tool.poc.poc_commands import get_prod_dir, setup_service_config

    project, _ = setup_service_config(str(WORKSPACE))
    return Path(get_prod_dir(str(WORKSPACE), project["name"]))
