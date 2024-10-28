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
from typing import Optional


class Example:
    """This class represents a standardized example folder structure in NVFlare."""

    def __init__(
        self,
        root: str,
        jobs_folder_in_example: str = "jobs",
        requirements: str = "requirements.txt",
        additional_python_path: Optional[str] = None,
        prepare_data_script: Optional[str] = None,
    ):
        """Constructor of Example.

        A standardized example folder looks like the following:

            .. code-block

                ./[example_root]
                    ./[jobs_folder_in_example]
                        ./job_name1
                        ./job_name2
                        ./job_name3
                    ./[requirements]
                    ./[prepare_data_script]

        For example:

            .. code-block

                ./cifar10-sim
                    ./jobs
                        ./cifar10_central
                        ./cifar10_fedavg
                        ./cifar10_fedopt
                        ...
                    ./requirements.txt
                    ./prepare_data.sh

        """
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError("Example's root directory does not exist.")

        self.name = os.path.basename(self.root)

        self.jobs_root_dir = os.path.join(self.root, jobs_folder_in_example)
        if not os.path.exists(self.jobs_root_dir):
            raise FileNotFoundError("Example's jobs root directory does not exist.")

        self.requirements_file = os.path.join(self.root, requirements)
        if not os.path.exists(self.requirements_file):
            raise FileNotFoundError("Example's requirements file does not exist.")

        self.additional_python_paths = [self.root]
        if additional_python_path is not None:
            if not os.path.exists(additional_python_path):
                raise FileNotFoundError(f"Additional python path ({additional_python_path}) does not exist")
            self.additional_python_paths.append(os.path.abspath(additional_python_path))

        if prepare_data_script is not None:
            prepare_data_script = os.path.join(self.root, prepare_data_script)
            if not os.path.exists(prepare_data_script):
                raise FileNotFoundError(f"Prepare_data_script ({prepare_data_script}) does not exist")
        self.prepare_data_script = prepare_data_script
