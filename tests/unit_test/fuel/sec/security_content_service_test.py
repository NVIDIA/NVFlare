# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentManager


def test_signed_commented_json_verifies_original_bytes(tmp_path):
    content = '// site configuration\n{"name": "α",}'.encode()
    path = tmp_path / "fed_client.json"
    path.write_bytes(content)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    manager = SecurityContentManager(str(tmp_path))
    manager.valid_config = True
    manager.public_key = key.public_key()
    manager.signature[path.name] = key.sign(
        content, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256()
    )
    assert manager.load_json(path.name) == ({"name": "α"}, LoadResult.OK)
    path.write_bytes(content.replace(b"site configuration", b"changed comment"))
    assert manager.load_json(path.name) == ({"name": "α"}, LoadResult.INVALID_SIGNATURE)
