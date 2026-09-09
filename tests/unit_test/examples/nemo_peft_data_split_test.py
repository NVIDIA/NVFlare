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

import importlib.util
import json
import os

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")


def _load_split_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "integration",
            "nemo",
            "examples",
            "peft",
            "data",
            "split_financial_phrase_data.py",
        )
    )
    spec = importlib.util.spec_from_file_location("nemo_peft_data_split", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_financial_phrase_split_is_fixed_hashed_and_sentence_disjoint(tmp_path):
    module = _load_split_module()
    labels = (" negative", " neutral", " positive")
    train_rows = [
        {"sentence": f"unique training sentence {index}", "label": labels[index % len(labels)]} for index in range(90)
    ]
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_rows(train, train_rows)
    _write_rows(validation, [{"sentence": "validation only", "label": " neutral"}])
    _write_rows(test, [{"sentence": "test only", "label": " positive"}])

    manifests = []
    for name in ("first", "second"):
        output = tmp_path / name
        module.split_data(str(train), str(output), 3, "site-", 0, 10.0, str(validation), str(test))
        with open(output / "split_manifest.json") as f:
            manifests.append(json.load(f))

    assert manifests[0]["sentence_level_disjoint"] is True
    assert [manifests[0]["files"][f"site-{idx}"]["sha256"] for idx in range(1, 4)] == [
        manifests[1]["files"][f"site-{idx}"]["sha256"] for idx in range(1, 4)
    ]
    assert sum(manifests[0]["files"][f"site-{idx}"]["rows"] for idx in range(1, 4)) == len(train_rows)


def test_financial_phrase_split_rejects_sentence_overlap(tmp_path):
    module = _load_split_module()
    labels = (" negative", " neutral", " positive")
    train_rows = [
        {"sentence": f"unique training sentence {index}", "label": labels[index % len(labels)]} for index in range(90)
    ]
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_rows(train, train_rows)
    _write_rows(validation, [{"sentence": train_rows[0]["sentence"], "label": " neutral"}])

    with pytest.raises(ValueError, match="Sentence-level dataset split overlap"):
        module.split_data(str(train), str(tmp_path / "split"), 3, "site-", 0, 10.0, str(validation), None)
