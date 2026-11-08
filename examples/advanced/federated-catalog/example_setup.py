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

"""Prepare file-backed sites using the same provisioned deployment setup as the service example."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location(
        "catalog_deployment_setup", ROOT.parent / "federated-catalog-services" / "example_setup.py"
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    setup.main(ROOT)
