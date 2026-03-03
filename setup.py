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
import importlib.util
import os
import shutil

from setuptools import find_packages, setup

def load_local_versioneer():
    root = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
    versioneer_path = os.path.join(root, "versioneer.py")
    spec = importlib.util.spec_from_file_location("nvflare_local_versioneer", versioneer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load versioneer from {versioneer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versioneer = load_local_versioneer()

# read the contents of your README file

versions = versioneer.get_versions()
base_version = os.environ.get("NVFL_BASE_VERSION")
if versions["error"]:
    today = datetime.date.today().timetuple()
    year = today[0] % 1000
    month = today[1]
    day = today[2]
    if base_version:
        version = f"{base_version}.dev{year:02d}{month:02d}{day:02d}"
    else:
        version = f"2.6.0.dev{year:02d}{month:02d}{day:02d}"
else:
    version = versions["version"]

release = os.environ.get("NVFL_RELEASE")
if release == "1":
    package_name = "nvflare"
else:
    package_name = "nvflare-nightly"


def package_files(
    root,
    starting,
):
    paths = []
    for path, directories, filenames in os.walk(os.path.join(root, starting)):
        rel_dir = os.path.relpath(path, root)
        for filename in filenames:
            paths.append(os.path.join(rel_dir, filename))
    return paths


def copy_package(src_dir, dst_dir):
    if os.path.isdir(src_dir):
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    for root, dirs, files in os.walk(dst_dir):
        for f in files:
            if f.endswith(".md"):
                os.remove(os.path.join(root, f))


def remove_dir(target_path):
    if target_path and os.path.isdir(target_path):
        shutil.rmtree(target_path)


extra_files = package_files(root="nvflare/dashboard/application", starting="static")
tmp_job_template_folder = "./nvflare/tool/job/templates"
copy_package(src_dir="job_templates", dst_dir=tmp_job_template_folder)
job_templates = package_files(root="nvflare/tool/job", starting="templates")


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
    package_data={
        "": ["*.yml", "*.html", "*.js", "poc.zip", "*.config", "*.conf"],
        "nvflare.dashboard.application": extra_files,
        "nvflare.tool.job": job_templates,
    },
    include_package_data=True,
)

remove_dir(target_path=tmp_job_template_folder)
