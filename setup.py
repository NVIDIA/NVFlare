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
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

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
    version = f"0.0.{year:02d}{month:02d}{day:02d}"
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
    description="Federated Learning Application Runtime Environment",
    url="https://github.com/NVIDIA/NVFlare",
    package_dir={"nvflare": "nvflare"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=["tests", "tests.*"],
    ),
    package_data={"": ["*.yml", "*.html", "poc.zip"]},
    zip_safe=True,
    license_files=("LICENSE",),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7,<3.9",
    install_requires=[
        "cryptography>=36.0.0",
        "Flask==2.1.2",
        "google-api-python-client==2.49.0",
        "grpcio==1.46.3",
        "gunicorn==20.1.0",
        "numpy",
        "protobuf==3.20.1",
        "psutil==5.9.1",
        "PyYAML==6.0",
        "six>=1.15.0",
        "tenseal==0.3.0",
        "msgpack==1.0.3",
        "docker>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "provision=nvflare.lighter.provision:main",
            "poc=nvflare.lighter.poc:main",
            "nvflare=nvflare.cli:main",
            "authz_preview=nvflare.fuel.hci.tools.authz_preview:main",
        ],
    },
)

os.remove(os.path.join(this_directory, "nvflare", "poc.zip"))
