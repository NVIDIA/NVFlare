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

import phe

SCALE_FACTOR = 10000000000000


def combine(g, h):
    return g * SCALE_FACTOR + h


def split(d):
    combined_g = d / SCALE_FACTOR
    g = int(round(combined_g, 0))
    h = d - g * SCALE_FACTOR
    return g, h


def generate_keys(key_length=1024):
    return phe.paillier.generate_paillier_keypair(n_length=key_length)


def _encode_encrypted_numbers(numbers):
    result = []
    for x in numbers:
        if isinstance(x, phe.paillier.EncryptedNumber):
            result.append((phe.util.int_to_base64(x.ciphertext()), x.exponent))
        else:
            result.append(x)
    return result


def encode_encrypted_numbers_to_str(numbers):
    return json.dumps(_encode_encrypted_numbers(numbers))


def encode_encrypted_data(pubkey, encrypted_numbers) -> str:
    result = {"key": {"n": phe.util.int_to_base64(pubkey.n)}, "nums": _encode_encrypted_numbers(encrypted_numbers)}
    return json.dumps(result)


def decode_encrypted_data(encoded: str):
    data = json.loads(encoded)
    k = data["key"]
    pubkey = phe.paillier.PaillierPublicKey(n=phe.util.base64_to_int(k["n"]))
    numbers = data["nums"]
    result = _decode_encrypted_numbers(pubkey, numbers)
    return pubkey, result


def decode_encrypted_numbers_from_str(pubkey, encoded: str):
    j = json.loads(encoded)
    return _decode_encrypted_numbers(pubkey, j)


def _decode_encrypted_numbers(pubkey, data):
    result = []
    for v in data:
        if isinstance(v, int):
            d = v
        else:
            d = phe.paillier.EncryptedNumber(pubkey, ciphertext=phe.util.base64_to_int(v[0]), exponent=int(v[1]))
        result.append(d)
    return result


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
