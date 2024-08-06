# Copyright (c) 2023, NVIDIA CORPORATION.
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

import datetime
import os

from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))

today = datetime.date.today().timetuple()
year = today[0] % 1000
month = today[1]
day = today[2]

release_package = find_packages(
    where=".",
    include=[
        "*",
    ],
    exclude=["tests", "tests.*"],
)

package_data = {"": ["*.yml", "*.config"], }

release = os.environ.get("NVFL_RELEASE")
version = os.environ.get("NVFL_VERSION")

if release == "1":
    package_dir = {"nvflare": "nvflare"}
    package_name = "nvflare-light"
else:
    package_dir = {"nvflare": "nvflare"}
    package_name = "nvflare-light-nightly"

setup(
    name=package_name,
    version=version,
    package_dir=package_dir,
    packages=release_package,
    package_data=package_data,
    include_package_data=True,
)
