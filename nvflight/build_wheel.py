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

import datetime
import os
import shutil
import subprocess

from prepare_setup import prepare_setup

import versioneer

versions = versioneer.get_versions()
if versions["error"]:
    today = datetime.date.today().timetuple()
    year = today[0] % 1000
    month = today[1]
    day = today[2]
    version = f"2.3.9.dev{year:02d}{month:02d}{day:02d}"
else:
    version = versions["version"]


def patch(setup_dir, patch_file):
    file_dir_path = os.path.abspath(os.path.dirname(__file__))
    cmd = ['git', 'apply', os.path.join(file_dir_path, patch_file)]
    try:
        subprocess.run(cmd, check=True, cwd=setup_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error to patch prepared files {e}")
        exit(1)

nvflight_setup_dir = "/tmp/nvflight_setup"
patch_file = "patch.diff"
# prepare
prepare_setup(nvflight_setup_dir)

patch(nvflight_setup_dir, patch_file)
# build wheel
dist_dir = os.path.join(nvflight_setup_dir, "dist")
if os.path.isdir(dist_dir):
    shutil.rmtree(dist_dir)

env = os.environ.copy()
env['NVFL_VERSION'] = version

cmd_str = "python setup.py -v sdist bdist_wheel"
cmd = cmd_str.split(" ")
try:
    subprocess.run(cmd, check=True, cwd=nvflight_setup_dir, env=env)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

results = []
for root, dirs, files in os.walk(dist_dir):
    result = [os.path.join(root, f) for f in files if f.endswith(".whl")]
    results.extend(result)

if not os.path.isdir("dist"):
    os.makedirs("dist", exist_ok=True)

if len(results) == 1:
    shutil.copy(results[0], os.path.join("dist", os.path.basename(results[0])))
else:
    print(f"something is not right, wheel files = {results}")

print(f"Setup dir {nvflight_setup_dir}")
shutil.rmtree(nvflight_setup_dir)
