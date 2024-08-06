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


import os
import shutil

exclude_extensions = [".md", ".rst", ".pyc", "__pycache__"]

nvflight_packages = {
    "nvflare": {
        "include": ["_version.py"],
        "exclude": ["*"]
    },
    "nvflare/apis": {
        "include": ["__init__.py", "fl_constant.py"],
        "exclude": ["*"]
    },
    "nvflare/app_common": {
        "include": ["__init__.py"],
        "exclude": ["*"]
    },
    "nvflare/app_common/decomposers": {
        "include": ["__init__.py", "numpy_decomposers.py"],
        "exclude": ["*"]
    },
    "nvflare/client": {
        "include": ["__init__.py"],
        "exclude": ["*"]
    },
    "nvflare/client/ipc": {
        "include": ["__init__.py", "defs.py", "ipc_agent.py"],
        "exclude": ["*"]
    },
    "nvflare/fuel": {
        "include": ["__init__.py"],
        "exclude": ["*"]
    },
    "nvflare/fuel/common": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/fuel/f3": {
        "include": ["__init__.py",
                     "comm_error.py",
                     "connection.py",
                     "endpoint.py",
                     "mpm.py",
                     "stats_pool.py",
                     "comm_config.py",
                     "communicator.py",
                     "message.py",
                     "stream_cell.py"
        ],
        "exclude": ["*"]
    },
    "nvflare/fuel/f3/cellnet": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/fuel/f3/drivers": {
        "include": ["*"],
        "exclude": ["grpc", "aio_grpc_driver.py", "aio_http_driver.py", "grpc_driver.py"]
    },
    "nvflare/fuel/f3/sfm": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/fuel/f3/streaming": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/fuel/hci": {
        "include": ["__init__.py", "security.py"],
        "exclude": ["*"]
    },
    "nvflare/fuel/utils": {
        "include": ["*"],
        "exclude": ["fobs"]
    },
    "nvflare/fuel/utils/fobs": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/fuel/utils/fobs/decomposers": {
        "include": ["*"],
        "exclude": []
    },
    "nvflare/security": {
        "include": ["__init__.py", "logging.py"],
        "exclude": ["*"]
    }
}


def should_exclude(str_value):
    return any(str_value.endswith(ext) for ext in exclude_extensions)


def package_selected_files(package_info: dict):
    if not package_info:
        return
    all_items = "*"
    results = {}

    for p, package_rule in package_info.items():
        include = package_rule["include"]
        exclude = package_rule["exclude"]
        paths = []
        for include_item in include:
            item_path = os.path.join(p, include_item)
            if all_items != include_item:
                if all_items in exclude:
                    # excluded everything except for included items
                    if os.path.isfile(item_path) and not should_exclude(item_path):
                        paths.append(item_path)
                elif include_item not in exclude:
                    paths.append(item_path)
            else:
                if all_items in exclude:
                    # excluded everything except for included items
                    if os.path.isfile(item_path):
                        paths.append(item_path)
                else:
                    # include everything in the package except excluded items
                    for root, dirs, files in os.walk(p):
                        if should_exclude(root) or os.path.basename(root) in exclude:
                            continue

                        for f in files:
                            if not should_exclude(f) and f not in exclude:
                                paths.append(os.path.join(root, f))
        results[p] = paths
    return results


def create_empty_file(file_path):
    try:
        with open(file_path, 'w'):
            pass  # This block is intentionally left empty
    except Exception as e:
        print(f"Error creating empty file: {e}")


def copy_files(package_paths: dict, target_dir: str):
    for p, paths in package_paths.items():
        for src_path in paths:
            dst_path = os.path.join(target_dir, src_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

    for p in package_paths:
        init_file_path = os.path.join(target_dir, p, "__init__.py")
        if not os.path.isfile(init_file_path):
            create_empty_file(init_file_path)


def prepare_setup(setup_dir: str):
    if os.path.isdir(setup_dir):
        shutil.rmtree(setup_dir)

    os.makedirs(setup_dir, exist_ok=True)
    nvflight_paths = package_selected_files(nvflight_packages)
    copy_files(nvflight_paths, setup_dir)

    src_files = [
        "setup.cfg",
        "README.md",
        "LICENSE",
         os.path.join("nvflight", "setup.py")
    ]

    for src in src_files:
        shutil.copy(src, os.path.join(setup_dir, os.path.basename(src)))
    
