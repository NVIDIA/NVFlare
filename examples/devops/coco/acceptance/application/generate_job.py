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

"""Generate a data/configuration-only job referencing preinstalled acceptance classes."""

import argparse
import json
from pathlib import Path

from tdx_acceptance import EXPECTED_VALUES, TASK_NAME, validate_nonce


def generate_job(output, nonce):
    nonce = validate_nonce(nonce)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = output / "app" / "config"
    config.mkdir(parents=True)
    documents = {
        output
        / "meta.json": {
            "name": "tdx-acceptance-" + nonce,
            "min_clients": 2,
            "mandatory_clients": list(EXPECTED_VALUES),
            "deploy_map": {"app": ["server", *EXPECTED_VALUES]},
            "resource_spec": {},
        },
        config
        / "config_fed_server.json": {
            "format_version": 2,
            "workflows": [
                {"id": "acceptance", "path": "tdx_acceptance.AcceptanceController", "args": {"nonce": nonce}}
            ],
            "components": [],
            "task_data_filters": [],
            "task_result_filters": [],
        },
        config
        / "config_fed_client.json": {
            "format_version": 2,
            "executors": [
                {
                    "tasks": [TASK_NAME],
                    "executor": {"path": "tdx_acceptance.AcceptanceExecutor", "args": {"nonce": nonce}},
                }
            ],
            "components": [],
            "task_data_filters": [],
            "task_result_filters": [],
        },
    }
    for path, document in documents.items():
        path.write_text(json.dumps(document, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nonce", required=True, help="fresh run nonce, e.g. uuid.uuid4().hex")
    args = parser.parse_args()
    generate_job(args.output, args.nonce)


if __name__ == "__main__":
    main()
