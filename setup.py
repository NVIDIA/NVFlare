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

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()


def load_local_versioneer():
    versioneer_path = os.path.join(ROOT_DIR, "versioneer.py")
    spec = importlib.util.spec_from_file_location("nvflare_local_versioneer", versioneer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load versioneer from {versioneer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_local_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versioneer = load_local_versioneer()
agent_skill_manifest = load_local_module(
    "nvflare_agent_skill_manifest", os.path.join(ROOT_DIR, "nvflare", "tool", "agent", "skill_manifest.py")
)

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


def _package_agent_skills_enabled():
    value = os.environ.get("NVFLARE_PACKAGE_AGENT_SKILLS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _no_skills_wheel_build_tag():
    return os.environ.get("NVFLARE_NO_SKILLS_WHEEL_BUILD_TAG", "1no_skills").strip()


extra_files = package_files(root="nvflare/dashboard/application", starting="static")
tmp_job_template_folder = "./nvflare/tool/job/templates"
copy_package(src_dir="job_templates", dst_dir=tmp_job_template_folder)
job_templates = package_files(root="nvflare/tool/job", starting="templates")
deploy_templates = package_files(root="nvflare/tool/deploy", starting="templates")
agent_skill_package_data = ["manifest.json", "*", "*/*", "*/*/*", "*/*/*/*", "*/*/*/*/*"]

cmdclass = versioneer.get_cmdclass()
_base_build_py = cmdclass["build_py"]


class AgentSkillsBuildPy(_base_build_py):
    def run(self):
        super().run()
        bundle_root = os.path.join(self.build_lib, "nvflare", "tool", "agent", "bundled_skills")
        if _package_agent_skills_enabled():
            agent_skill_manifest.copy_released_skills_to_bundle(
                os.path.join(ROOT_DIR, "skills"),
                bundle_root,
                nvflare_version=version,
            )
        else:
            agent_skill_manifest.write_empty_skill_bundle(
                bundle_root,
                nvflare_version=version,
            )


cmdclass["build_py"] = AgentSkillsBuildPy


def _agent_skills_bdist_wheel_cmd():
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        try:
            from wheel.bdist_wheel import bdist_wheel
        except ImportError:
            return None

    class AgentSkillsBdistWheel(bdist_wheel):
        def finalize_options(self):
            if not _package_agent_skills_enabled() and not self.build_number:
                self.build_number = _no_skills_wheel_build_tag()
            super().finalize_options()

    return AgentSkillsBdistWheel


_bdist_wheel_cmd = _agent_skills_bdist_wheel_cmd()
if _bdist_wheel_cmd is not None:
    cmdclass["bdist_wheel"] = _bdist_wheel_cmd


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
        exclude=["tests", "tests.*"],
    ),
    package_data={
        "": ["*.yml", "*.yaml", "*.tpl", "*.html", "*.js", "poc.zip", "*.config", "*.conf"],
        "nvflare.dashboard.application": extra_files,
        "nvflare.tool.job": job_templates,
        "nvflare.tool.deploy": deploy_templates,
        "nvflare.tool.agent.bundled_skills": agent_skill_package_data,
    },
    include_package_data=True,
)

remove_dir(target_path=tmp_job_template_folder)
