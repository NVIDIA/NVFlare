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

import threading

data_store = dict()
data_store["SP"] = dict()
data_store_lock = threading.Lock()


def _primary_key(sp):
    return f'{sp["project"]}/{sp["sp_end_point"]}'


def get_all_sp(project):
    with data_store_lock:
        sp_list = [v for v in data_store["SP"].values() if v["project"] == project]
    return sp_list


def get_primary_sp(project):
    psp = {}
    with data_store_lock:
        for _, sp in data_store["SP"].items():
            if sp["primary"] and sp["project"] == project:
                psp = sp
                break
    return psp


def update_sp(sp):
    with data_store_lock:
        key = _primary_key(sp)
        existing_sp = data_store["SP"].get(key)
        if existing_sp:
            existing_sp.update(sp)
            data_store["SP"][key] = existing_sp
        else:
            data_store["SP"][key] = sp


def get_sp_by(predicate: dict):
    result = {}
    with data_store_lock:
        for sp in data_store["SP"].values():
            if all(sp[k] == predicate[k] for k in predicate):
                result = sp
                break
    return result
