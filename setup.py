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

import datetime
import importlib.util
import os
import shutil

from setuptools import find_packages, setup

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()


def load_local_module(name, relative_path):
    module_path = os.path.join(ROOT_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versioneer = load_local_module("nvflare_local_versioneer", "versioneer.py")

# read the contents of your README file

versions = versioneer.get_versions()
if versions["error"]:
    base_version = os.environ.get("NVFL_BASE_VERSION")
    if not base_version:
        raise RuntimeError(
            "Versioneer could not determine the NVFlare package version. "
            "Build from a source tree with Git or embedded Versioneer metadata, "
            "or set NVFL_BASE_VERSION explicitly."
        )
    date_suffix = datetime.date.today().strftime("%y%m%d")
    version = f"{base_version}.dev{date_suffix}"
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
deploy_templates = package_files(root="nvflare/tool/deploy", starting="templates")
example_source_folder = "./examples/hello-world/hello-pt"
tmp_example_folder = "./nvflare/tool/examples/data"
generated_example_data = os.path.isdir(example_source_folder)
example_definitions = load_local_module("nvflare_example_definitions", "nvflare/tool/examples/__init__.py")
hello_pt_files = example_definitions.HELLO_PT_FILES
if generated_example_data:
    remove_dir(target_path=tmp_example_folder)
    for filename in hello_pt_files:
        target = os.path.join(tmp_example_folder, "hello-pt", filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(example_source_folder, filename), target)
example_files = [os.path.join("data", "hello-pt", filename) for filename in hello_pt_files]

cmdclass = versioneer.get_cmdclass()


setup(
    name=package_name,
    version=version,
    cmdclass=cmdclass,
    package_dir={"nvflare": "nvflare"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=["tests", "tests.*", "dev_tools", "dev_tools.*"],
    ),
    package_data={
        "": ["*.yml", "*.yaml", "*.tpl", "*.html", "*.js", "poc.zip", "*.config", "*.conf"],
        "nvflare.dashboard.application": extra_files,
        "nvflare.tool.job": job_templates,
        "nvflare.tool.deploy": deploy_templates,
        "nvflare.tool.examples": example_files,
        "nvflare.tool.recipe": ["recipe_catalog.json"],
    },
    include_package_data=True,
)

remove_dir(target_path=tmp_job_template_folder)
if generated_example_data:
    remove_dir(target_path=tmp_example_folder)
