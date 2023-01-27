# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import shutil

from setuptools import find_packages, setup

import versioneer

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))

if os.path.exists(os.path.join(this_directory, "nvflare", "poc.zip")):
    os.remove(os.path.join(this_directory, "nvflare", "poc.zip"))
shutil.make_archive(base_name="poc", format="zip", root_dir=os.path.join(this_directory, "nvflare"), base_dir="poc")
shutil.move("poc.zip", os.path.join(this_directory, "nvflare", "poc.zip"))

versions = versioneer.get_versions()
if versions["error"]:
    today = datetime.date.today().timetuple()
    year = today[0] % 1000
    month = today[1]
    day = today[2]
    version = f"2.2.4.dev{year:02d}{month:02d}{day:02d}"
else:
    version = versions["version"]

release = os.environ.get("NVFL_RELEASE")
if release == "1":
    package_name = "nvflare"
else:
    package_name = "nvflare-nightly"

setup(
    name=package_name,
    version=version,
    cmdclass=versioneer.get_cmdclass(),
    package_dir={"nvflare": "nvflare"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=["tests", "tests.*"],
    ),
    package_data={"": ["*.yml", "*.html", "poc.zip", "*.config"]},
)

os.remove(os.path.join(this_directory, "nvflare", "poc.zip"))
