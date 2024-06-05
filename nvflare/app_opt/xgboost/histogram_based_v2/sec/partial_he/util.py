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
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import hexlify, unhexlify

# ipcl_python is not a required dependency. The import error causes unit test failure so make it optional
try:
    import ipcl_python
    from ipcl_python import PaillierEncryptedNumber as EncryptedNumber
    from ipcl_python.ipcl_python import BNUtils, ipclCipherText

    ipcl_imported = True
except Exception:
    ipcl_imported = False

SCALE_FACTOR = 10000000000000
ENABLE_DJN = True


def generate_keys(n_length=1024):
    return ipcl_python.PaillierKeypair.generate_keypair(n_length=n_length, enable_DJN=ENABLE_DJN)


def encrypt_number(pubkey, ciphertext, exponent):
    return EncryptedNumber(pubkey, ciphertext, [exponent], 1)


def create_pub_key(key, n_length=1024):
    return ipcl_python.PaillierPublicKey(key=key, n_length=n_length, enable_DJN=ENABLE_DJN)


def ciphertext_to_int(d):
    cifer = d.ciphertextBN()
    return BNUtils.BN2int(cifer[0])


def int_to_ciphertext(d, pubkey):
    return ipclCipherText(pubkey.pubkey, BNUtils.int2BN(d))


def get_exponent(d):
    return d.exponent(idx=0)


# base64 utils from jwcrypto
def base64url_encode(payload):
    if not isinstance(payload, bytes):
        payload = payload.encode("utf-8")
    encode = urlsafe_b64encode(payload)
    return encode.decode("utf-8").rstrip("=")


def base64url_decode(payload):
    l = len(payload) % 4
    if l == 2:
        payload += "=="
    elif l == 3:
        payload += "="
    elif l != 0:
        raise ValueError("Invalid base64 string")
    return urlsafe_b64decode(payload.encode("utf-8"))


def base64_to_int(source):
    return int(hexlify(base64url_decode(source)), 16)


def int_to_base64(source):
    assert source != 0
    I = hex(source).rstrip("L").lstrip("0x")
    return base64url_encode(unhexlify((len(I) % 2) * "0" + I))


def combine(g, h):
    return g * SCALE_FACTOR + h


def split(d):
    combined_g = d / SCALE_FACTOR
    g = int(round(combined_g, 0))
    h = d - g * SCALE_FACTOR
    return g, h


def _encode_encrypted_numbers(numbers):
    result = []
    for x in numbers:
        if isinstance(x, EncryptedNumber):
            result.append((int_to_base64(ciphertext_to_int(x)), get_exponent(x)))
        else:
            result.append(x)
    return result


def encode_encrypted_numbers_to_str(numbers):
    return json.dumps(_encode_encrypted_numbers(numbers))


def encode_encrypted_data(pubkey, encrypted_numbers) -> str:
    result = {"key": {"n": int_to_base64(pubkey.n)}, "nums": _encode_encrypted_numbers(encrypted_numbers)}
    return json.dumps(result)


def decode_encrypted_data(encoded: str, n_length=1024):
    data = json.loads(encoded)
    pubkey = create_pub_key(key=base64_to_int(data["key"]["n"]), n_length=n_length)
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
            d = encrypt_number(
                pubkey, ciphertext=int_to_ciphertext(base64_to_int(v[0]), pubkey=pubkey), exponent=int(v[1])
            )
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
