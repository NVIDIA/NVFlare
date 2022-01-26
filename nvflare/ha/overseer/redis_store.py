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

import redis

redis_db = redis.Redis(charset="utf-8", decode_responses=True)
SP = "SP"


def do_refresh():
    redis_db.flushdb()


def _boolean(sp):
    if "primary" in sp:
        if isinstance(sp["primary"], str):
            sp["primary"] = sp["primary"] == "True"
        elif isinstance(sp["primary"], bool):
            sp["primary"] = str(sp["primary"])
    return sp


def _primary_key(sp):
    return f'{sp["project"]}/{sp["sp_end_point"]}'


def get_all_sp(project):
    sp_list = list()
    for sp_key in redis_db.smembers(SP):
        if project in sp_key:
            sp_list.append(_boolean(redis_db.hgetall(sp_key)))
    return sp_list


def get_primary_sp(project):
    psp = {}
    for sp_key in redis_db.smembers(SP):
        if project in sp_key:
            sp = redis_db.hgetall(sp_key)
            if sp["primary"] == "True":
                psp = sp
                break
    psp = _boolean(psp)
    return psp


def update_sp(sp):
    key = _primary_key(sp)
    sp = _boolean(sp)
    existing_sp = redis_db.hgetall(key)
    if existing_sp:
        existing_sp.update(sp)
        redis_db.hset(key, mapping=existing_sp)
    else:
        redis_db.hset(key, mapping=sp)
        redis_db.sadd(SP, key)


def get_sp_by(predicate: dict):
    result = {}
    predicate = _boolean(predicate)
    for sp_key in redis_db.smembers(SP):
        sp = redis_db.hgetall(sp_key)
        if all(sp[k] == predicate[k] for k in predicate):
            result = sp
            break
    result = _boolean(result)
    return result
