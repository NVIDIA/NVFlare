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

import os
import shutil
import datetime

from setuptools import find_packages, setup

import versioneer
<<<<<<< HEAD
=======

>>>>>>> upstream/main
# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

<<<<<<< HEAD
# with open(os.path.join(this_directory, "nvflare", "__init__.py"), encoding="utf-8") as f:
#     for line in f.readlines():
#         if "__version__" in line:
#             init_version = line.split("=")[1].strip().strip('"')
# 
# nvfl_version = os.environ.get("NVFL_VERSION", init_version)
# yymmdd = datetime.today().strftime("%y%m%d")
# nvfl_nightly_version = f"{nvfl_version}.dev{yymmdd}"
# 
# if os.environ.get("NVFL_RELEASE"):
#     package_name = "nvflare"
#     version = nvfl_version
# else:
#     package_name = "nvflare-nightly"
#     version = nvfl_nightly_version
# 

if os.path.exists(os.path.join(this_directory, "nvflare", "poc.zip")):
    os.remove(os.path.join(this_directory, "nvflare", "poc.zip"))
shutil.make_archive(base_name="poc", format="zip", root_dir=os.path.join(this_directory, "nvflare"), base_dir="poc")
shutil.move("poc.zip", os.path.join(this_directory, "nvflare", "poc.zip"))
package_name = "nvflare"

setup(
    name=package_name,
    version=versioneer.get_version(),
=======
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
>>>>>>> upstream/main
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
<<<<<<< HEAD
    python_requires=">=3.7",
    # install_requires=list(pkutils.parse_requirements("requirements.txt")),
    install_requires=["PyYAML", "psutil", "numpy", "grpcio", "google-api-python-client", "cryptography"],
    extras_require={"HE": ["tenseal==0.3.0"]},
=======
    python_requires=">=3.7,<3.9",
    install_requires=[
        "PyYAML",
        "psutil",
        "numpy",
        "grpcio",
        "google-api-python-client",
        "cryptography>=36.0.0",
        "tenseal==0.3.0",
        "gunicorn",
        "flask",
    ],
>>>>>>> upstream/main
    entry_points={
        "console_scripts": [
            "provision=nvflare.lighter.provision:main",
            "poc=nvflare.lighter.poc:main",
            "authz_preview=nvflare.fuel.hci.tools.authz_preview:main",
        ],
    },
)

os.remove(os.path.join(this_directory, "nvflare", "poc.zip"))
