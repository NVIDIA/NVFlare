# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

release = os.environ.get("MONAI_NVFL_RELEASE")
if release == "1":
    package_name = "monai-nvflare"
    version = "0.2.9"
else:
    package_name = "monai-nvflare-nightly"
    today = datetime.date.today().timetuple()
    year = today[0] % 1000
    month = today[1]
    day = today[2]
    version = f"0.2.9.{year:02d}{month:02d}{day:02d}"

setup(
    name=package_name,
    version=version,
    description="MONAI NVIDIA FLARE integration",
    url="https://github.com/NVIDIA/NVFlare",
    package_dir={"monai_nvflare": "monai_nvflare"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=["tests", "tests.*"],
    ),
    license_files=("LICENSE",),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8,<3.11",
    install_requires=["monai>=1.3.1", "nvflare~=2.5.0rc"],
)
