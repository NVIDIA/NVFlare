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

from cryptography.exceptions import InvalidKey, InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HASH_LENGTH = 4  # Adjustable to avoid collision
NONCE_LENGTH = 16  # For AES, this is 128 bits (i.e. block size)
KEY_LENGTH = 32  # AES 256.  Choose from 16, 24, 32
HEADER_LENGTH = HASH_LENGTH + NONCE_LENGTH
PADDING_LENGTH = NONCE_LENGTH * 8  # in bits
KEY_ENC_LENGTH = 256
SIGNATURE_LENGTH = 256


def get_hash(value):
    hash = hashes.Hash(hashes.SHA256())
    hash.update(value)
    return hash.finalize()


class SessionKeyUnavailable(Exception):
    pass


class InvalidCertChain(Exception):
    pass


class SessionKeyManager:
    def __init__(self, root_ca):
        self.key_hash_dict = dict()
        self.root_ca = root_ca
        self.root_ca_pub_key = root_ca.public_key()

    def validate_cert_chain(self, cert):
        self.root_ca_pub_key.verify(
            cert.signature, cert.tbs_certificate_bytes, padding.PKCS1v15(), cert.signature_hash_algorithm
        )

    def key_request(self, remote_cert, local_cert, local_pri_key):
        session_key = os.urandom(KEY_LENGTH)
        signature = local_pri_key.sign(
            data=session_key,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            algorithm=hashes.SHA256(),
        )
        try:
            self.validate_cert_chain(remote_cert)
        except InvalidSignature:
            return False

        remote_pub_key = remote_cert.public_key()
        key_enc = remote_pub_key.encrypt(
            session_key,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        self.key_hash_dict[get_hash(session_key)[-HASH_LENGTH:]] = session_key
        key_response = key_enc + signature
        return key_response

    def process_key_response(self, remote_cert, local_cert, local_pri_key, key_response):
        key_enc, signature = key_response[:KEY_ENC_LENGTH], key_response[KEY_ENC_LENGTH:]
        try:
            session_key = local_pri_key.decrypt(
                key_enc,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
            self.validate_cert_chain(remote_cert)
            public_key = remote_cert.public_key()
            public_key.verify(
                signature,
                session_key,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            self.key_hash_dict[get_hash(session_key)[-HASH_LENGTH:]] = session_key
        except (InvalidKey, InvalidSignature):
            return False
        return True

    def key_available(self):
        return bool(self.key_hash_dict)

    def get_key(self, key_hash):
        return self.key_hash_dict.get(key_hash)

    def get_latest_key(self):
        try:
            k, last_value = _, self.key_hash_dict[k] = self.key_hash_dict.popitem()
        except KeyError as e:
            raise SessionKeyUnavailable("No session key established yet")
        return last_value


class CellCipher:
    def __init__(self, session_key_manager: SessionKeyManager):
        self.session_key_manager = session_key_manager

    def encrypt(self, message):
        key = self.session_key_manager.get_latest_key()
        key_hash = get_hash(key)
        nonce = os.urandom(NONCE_LENGTH)
        cipher = Cipher(algorithms.AES(key), modes.CBC(nonce))
        encryptor = cipher.encryptor()
        padder = sym_padding.PKCS7(PADDING_LENGTH).padder()
        padded_data = padder.update(message) + padder.finalize()
        ct = nonce + key_hash[-HASH_LENGTH:] + encryptor.update(padded_data) + encryptor.finalize()
        return ct

    def decrypt(self, message):
        nonce, key_hash, message = (
            message[:NONCE_LENGTH],
            message[NONCE_LENGTH:HEADER_LENGTH],
            message[HEADER_LENGTH:],
        )
        key = self.session_key_manager.get_key(key_hash)
        if key is None:
            raise SessionKeyUnavailable("No session key found for received message")
        cipher = Cipher(algorithms.AES(key), modes.CBC(nonce))
        decryptor = cipher.decryptor()
        plain_text = decryptor.update(message)
        plain_text = plain_text + decryptor.finalize()
        unpadder = sym_padding.PKCS7(PADDING_LENGTH).unpadder()
        data = unpadder.update(plain_text) + unpadder.finalize()
        return data
