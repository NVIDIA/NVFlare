# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import shutil
from typing import List

from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.study_manager_spec import Study


def generate_meta(
    job_name: str,
    clients: List[str],
    study_name: str = Study.DEFAULT_STUDY_NAME
):
    resource_spec = {c: {"gpu": 1} for c in clients}
    deploy_map = {job_name: ["server"] + clients}
    meta = {
        "name": job_name,
        JobMetaKey.STUDY_NAME: study_name,
        JobMetaKey.RESOURCE_SPEC: resource_spec,
        JobMetaKey.DEPLOY_MAP: deploy_map,
    }
    return meta


def generate_job_dir_for_single_app_job(app_root_folder: str, app_name: str, clients: List[str], destination: str):
    app_folder = os.path.join(app_root_folder, app_name)
    if not os.path.exists(app_folder):
        raise RuntimeError(f"App folder {app_folder} does not exist.")
    if not os.path.isdir(app_folder):
        raise RuntimeError(f"App folder {app_folder} is not a folder.")

    job_folder = os.path.join(destination, app_name)
    if os.path.exists(job_folder):
        shutil.rmtree(job_folder)
    os.makedirs(job_folder)
    shutil.copytree(app_folder, os.path.join(job_folder, app_name))

    meta = generate_meta(job_name=app_name, clients=clients)
    with open(os.path.join(job_folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return job_folder


# def main():
#     app_root_folder = "./apps"
#     app_name = "pt"
#     clients = ["client_0", "client_1"]
#     destination = "./jobs"
#     generate_job_dir_for_single_app_job(app_root_folder, app_name, clients, destination)
#
#
# if __name__ == "__main__":
#     main()
