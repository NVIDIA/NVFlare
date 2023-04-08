# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from argparse import ArgumentParser

import yaml

from tests.integration_test.src import generate_test_config_yaml_for_example, read_yaml
from tests.integration_test.src.example import Example


def _construct_example_from_registry(registry: dict, examples_root: str) -> Example:
    example_root = os.path.join(examples_root, registry["root"])
    jobs_folder_in_example = registry.get("jobs_folder_in_example", "jobs")
    requirements_in_example = registry.get("requirements", "requirements.txt")
    additional_python_path = registry.get("additional_python_path")
    if additional_python_path is not None and not os.path.isabs(additional_python_path):
        additional_python_path = os.path.join(examples_root, additional_python_path)
    return Example(
        root=example_root,
        jobs_folder_in_example=jobs_folder_in_example,
        requirements=requirements_in_example,
        additional_python_path=additional_python_path,
        prepare_data_script=registry.get("prepare_data_script"),
    )


def main():
    parser = ArgumentParser("Generate all test configs")
    parser.add_argument(
        "--example_test_registry",
        default="example_registry.yml",
        type=str,
        help="a yaml file that specifies information needed to generate integration test's config for examples",
    )
    args = parser.parse_args()

    all_output_yamls = []

    # generate individual test yaml files
    example_list = read_yaml(args.example_test_registry)
    examples_root = example_list["examples_root"]
    for example_registry in example_list["examples"]:
        if "root" not in example_registry:
            print(f"Missing root attribute in registry: {example_registry}")
            continue
        try:
            example = _construct_example_from_registry(example_registry, examples_root)
            output_yamls = generate_test_config_yaml_for_example(example=example)
            all_output_yamls.extend(output_yamls)
        except FileNotFoundError as e:
            print(f"Skip invalid example entry ({example_registry}): {e}")
            continue

    # generate overall test config yaml
    test_config = {"test_configs": {"auto": all_output_yamls}}
    with open("auto_test_configs.yml", "w") as yaml_file:
        yaml.dump(test_config, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    main()
