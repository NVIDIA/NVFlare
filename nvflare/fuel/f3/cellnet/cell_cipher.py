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
from cryptography.hazmat.primitives import asymmetric, ciphers, hashes, padding
from cryptography.x509 import Certificate

HASH_LENGTH = 4  # Adjustable to avoid collision
NONCE_LENGTH = 16  # For AES, this is 128 bits (i.e. block size)
KEY_LENGTH = 32  # AES 256.  Choose from 16, 24, 32
HEADER_LENGTH = HASH_LENGTH + NONCE_LENGTH
PADDING_LENGTH = NONCE_LENGTH * 8  # in bits
KEY_ENC_LENGTH = 256
SIGNATURE_LENGTH = 256
SIMPLE_HEADER_LENGTH = NONCE_LENGTH + KEY_ENC_LENGTH + SIGNATURE_LENGTH


def get_hash(value):
    hash = hashes.Hash(hashes.SHA256())
    hash.update(value)
    return hash.finalize()


class SessionKeyUnavailable(Exception):
    pass


class InvalidCertChain(Exception):
    pass


def _asym_enc(k, m):
    return k.encrypt(
        m,
        asymmetric.padding.OAEP(
            mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )


def _asym_dec(k, m):
    return k.decrypt(
        m,
        asymmetric.padding.OAEP(
            mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )


def _sign(k, m):
    return k.sign(
        data=m,
        padding=asymmetric.padding.PSS(
            mgf=asymmetric.padding.MGF1(hashes.SHA256()),
            salt_length=asymmetric.padding.PSS.MAX_LENGTH,
        ),
        algorithm=hashes.SHA256(),
    )


def _verify(k, m, s):

    if not isinstance(m, bytes):
        m = bytes(m)

    if not isinstance(s, bytes):
        s = bytes(s)

    k.verify(
        s,
        m,
        asymmetric.padding.PSS(
            mgf=asymmetric.padding.MGF1(hashes.SHA256()), salt_length=asymmetric.padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )


def _sym_enc(k: bytes, n: bytes, m: bytes):
    cipher = ciphers.Cipher(ciphers.algorithms.AES(k), ciphers.modes.CBC(n))
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(PADDING_LENGTH).padder()
    padded_data = padder.update(m) + padder.finalize()
    return encryptor.update(padded_data) + encryptor.finalize()


def _sym_dec(k: bytes, n: bytes, m: bytes):
    cipher = ciphers.Cipher(ciphers.algorithms.AES(k), ciphers.modes.CBC(n))
    decryptor = cipher.decryptor()
    plain_text = decryptor.update(m)
    plain_text = plain_text + decryptor.finalize()
    unpadder = padding.PKCS7(PADDING_LENGTH).unpadder()
    return unpadder.update(plain_text) + unpadder.finalize()


class SessionKeyManager:
    def __init__(self, root_ca):
        self.key_hash_dict = dict()
        self.root_ca = root_ca
        self.root_ca_pub_key = root_ca.public_key()

    def validate_cert_chain(self, cert):
        self.root_ca_pub_key.verify(
            cert.signature, cert.tbs_certificate_bytes, asymmetric.padding.PKCS1v15(), cert.signature_hash_algorithm
        )

    def key_request(self, remote_cert, local_cert, local_pri_key):
        session_key = os.urandom(KEY_LENGTH)
        signature = _sign(local_pri_key, session_key)
        try:
            self.validate_cert_chain(remote_cert)
        except InvalidSignature:
            return False

        remote_pub_key = remote_cert.public_key()
        key_enc = _asym_enc(remote_pub_key, session_key)
        self.key_hash_dict[get_hash(session_key)[-HASH_LENGTH:]] = session_key
        key_response = key_enc + signature
        return key_response

    def process_key_response(self, remote_cert, local_cert, local_pri_key, key_response):
        key_enc, signature = key_response[:KEY_ENC_LENGTH], key_response[KEY_ENC_LENGTH:]
        try:
            session_key = _asym_dec(local_pri_key, key_enc)
            self.validate_cert_chain(remote_cert)
            public_key = remote_cert.public_key()
            _verify(public_key, session_key, signature)
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


class SimpleCellCipher:
    def __init__(self, root_ca: Certificate, pri_key: asymmetric.rsa.RSAPrivateKey, cert: Certificate):
        self._root_ca = root_ca
        self._root_ca_pub_key = root_ca.public_key()
        self._pri_key = pri_key
        self._cert = cert
        self._pub_key = cert.public_key()
        self._validate_cert_chain(self._cert)
        self._cached_enc = dict()
        self._cached_dec = dict()

    def _validate_cert_chain(self, cert: Certificate):
        self._root_ca_pub_key.verify(
            cert.signature, cert.tbs_certificate_bytes, asymmetric.padding.PKCS1v15(), cert.signature_hash_algorithm
        )

    def encrypt(self, message: bytes, target_cert: Certificate):
        cert_hash = hash(target_cert)
        secret = self._cached_enc.get(cert_hash)
        if secret is None:
            self._validate_cert_chain(target_cert)
            key = os.urandom(KEY_LENGTH)
            remote_pub_key = target_cert.public_key()
            key_enc = _asym_enc(remote_pub_key, key)
            signature = _sign(self._pri_key, key_enc)
            self._cached_enc[cert_hash] = (key, key_enc, signature)
        else:
            (key, key_enc, signature) = secret
        nonce = os.urandom(NONCE_LENGTH)
        ct = nonce + key_enc + signature + _sym_enc(key, nonce, message)
        return ct

    def decrypt(self, message: bytes, origin_cert: Certificate):
        nonce, key_enc, signature = (
            message[:NONCE_LENGTH],
            message[NONCE_LENGTH : NONCE_LENGTH + KEY_ENC_LENGTH],
            message[NONCE_LENGTH + KEY_ENC_LENGTH : SIMPLE_HEADER_LENGTH],
        )

        if not isinstance(key_enc, bytes):
            key_enc = bytes(key_enc)

        key_hash = hash(key_enc)
        dec = self._cached_dec.get(key_hash)
        if dec is None:
            self._validate_cert_chain(origin_cert)
            public_key = origin_cert.public_key()
            _verify(public_key, key_enc, signature)
            key = _asym_dec(self._pri_key, key_enc)
            self._cached_dec[key_hash] = key
        else:
            key = dec
        return _sym_dec(key, nonce, message[SIMPLE_HEADER_LENGTH:])
