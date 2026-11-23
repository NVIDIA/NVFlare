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


"""Store standalone ODCS JSON files and atomically publish the index that names them."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from nvflare.app_common.fdn.catalog.common.odcs import MAX_CONTRACT_BYTES
from nvflare.app_common.fdn.catalog.common.payload import check_catalog_results


def _replace(path, contents):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_index(path, snapshot):
    path = Path(path)
    folder = path.parent / "contracts"
    if path.parent.is_symlink() or folder.is_symlink():
        raise ValueError("unsafe catalog directory")
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in snapshot["datasets"]:
        contents = json.dumps(item["contract"], ensure_ascii=False, allow_nan=False).encode()
        filename = hashlib.sha256(contents).hexdigest() + ".odcs.json"
        _replace(folder / filename, contents)
        entries.append({"site": item["site"], "dataset": item["dataset"], "file": "contracts/" + filename})
    index = {**snapshot, "datasets": entries}
    _replace(path, json.dumps(index, indent=2, allow_nan=False).encode())


def read_index(path):
    if path.is_symlink() or path.parent.is_symlink() or (path.parent / "contracts").is_symlink():
        raise ValueError("unsafe catalog index")
    index = check_catalog_results(json.loads(path.read_text(encoding="utf-8")))
    entries = []
    for item in index["datasets"]:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key] for key in ("site", "dataset", "file")
        ):
            raise ValueError("invalid index entry")
        match = re.fullmatch(r"contracts/([a-f0-9]{64})\.odcs\.json", item["file"])
        if not match:
            raise ValueError("invalid contract path")
        contract = path.parent / item["file"]
        if contract.is_symlink() or not contract.is_file():
            raise ValueError("invalid contract file")
        with contract.open("rb") as stream:
            contents = stream.read(MAX_CONTRACT_BYTES + 1)
        if len(contents) > MAX_CONTRACT_BYTES or hashlib.sha256(contents).hexdigest() != match[1]:
            raise ValueError("invalid contract digest or size")
        entries.append({"site": item["site"], "dataset": item["dataset"], "contract": json.loads(contents)})
    return {**index, "datasets": entries}
