# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import sys


def get_security_handler() -> dict:
    return json.loads(
        """
        {
            "id": "security_handler",
            "path": "keycloak_security_handler.CustomSecurityHandler"
        }
        """
    )


def add_components_to_json(
    input_file_path, output_file_path, site: str, receiving: bool = False, streaming_to_server: bool = False
):
    try:
        with open(input_file_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Unable to read or parse JSON file: {input_file_path}")
        return

    new_components = [get_security_handler()]

    # Append new components to the list
    data["components"].extend(new_components)

    # Write the updated JSON back to the file
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully generate file: '{output_file_path}'.")


if __name__ == "__main__":

    site_name = sys.argv[1]
    project_root_dir = sys.argv[2]

    print(site_name, project_root_dir)

    input_file_path = os.path.join(project_root_dir, site_name, "local", "resources.json.default")
    output_file_path = os.path.join(project_root_dir, site_name, "local", "resources.json")

    add_components_to_json(input_file_path, output_file_path, site_name)
