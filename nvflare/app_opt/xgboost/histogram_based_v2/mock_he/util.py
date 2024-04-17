# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

SHIFT_FACTOR = 10000000000000


def combine(g, h):
    return g * SHIFT_FACTOR + h


def split(d):
    combined_g = d / SHIFT_FACTOR
    g = int(round(combined_g, 0))
    h = d - g * SHIFT_FACTOR
    return g, h


def generate_keys(key_length=1024):
    return "dummy_public_key", "dummy_private_key"


def _encode_encrypted_numbers(numbers):
    return numbers


def encode_encrypted_numbers_to_str(numbers):
    return json.dumps(_encode_encrypted_numbers(numbers))


def encode_encrypted_data(pubkey, encrypted_numbers) -> str:
    result = {"key": {"n": "dummy_key"}, "nums": _encode_encrypted_numbers(encrypted_numbers)}
    return json.dumps(result)


def decode_encrypted_data(encoded: str):
    data = json.loads(encoded)
    pubkey = data["key"]
    numbers = data["nums"]
    result = _decode_encrypted_numbers(pubkey, numbers)
    return pubkey, result


def decode_encrypted_numbers_from_str(pubkey, encoded: str):
    j = json.loads(encoded)
    return _decode_encrypted_numbers(pubkey, j)


def _decode_encrypted_numbers(pubkey, data):
    return data


def encode_feature_aggregations(aggrs: list):
    return json.dumps(aggrs)


def decode_feature_aggregations(pubkey, encoded: str):
    result = []
    aggrs = json.loads(encoded)
    for aggr in aggrs:
        feature_id, gid, encoded_nums_str = aggr
        encrypted_numbers = decode_encrypted_numbers_from_str(pubkey, encoded_nums_str)
        result.append((feature_id, gid, encrypted_numbers))
    return result
