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
import sys
import types
from collections import Counter
from pathlib import Path

import pytest


def _load_prepare_data_module():
    module_path = Path(__file__).parents[1] / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("evo2_prepare_data", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_data = _load_prepare_data_module()


def _record(index, label=None, chromosome="chr1", start=None, sequence=None, task="splice_sites_all"):
    label = index % 3 if label is None else label
    start = index * 1_000 if start is None else start
    sequence = f"ACGT{index:05d}" if sequence is None else sequence
    return {
        "sequence": sequence,
        "name": f"{chromosome}:{start}-{start + 600}|{label}",
        "label": label,
        "task": task,
    }


def _record_names(records):
    return [record["name"] for record in records]


def test_module_import_does_not_require_optional_data_packages():
    assert "datasets" not in prepare_data.__dict__
    assert "sklearn" not in prepare_data.__dict__


def test_source_loader_pins_revision_and_filters_splice_site_task(monkeypatch):
    captured = {}
    other_task = _record(100, task="enhancers")
    source_train = [_record(0), other_task]
    source_test = [_record(1, chromosome="chr2")]
    monkeypatch.setattr(prepare_data, "SOURCE_SPLIT_COUNTS", {"train": 1, "test": 1})
    monkeypatch.setattr(prepare_data, "SOURCE_SEQUENCE_LENGTH", len(source_train[0]["sequence"]))
    monkeypatch.setattr(prepare_data, "SOURCE_LABELS", frozenset({0, 1}))

    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(dataset_id, **kwargs):
        captured["dataset_id"] = dataset_id
        captured.update(kwargs)
        return {"train": source_train, "test": source_test}

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    train_records, test_records = prepare_data.load_source_records(cache_dir="/tmp/hf-cache")

    assert train_records == [_record(0)]
    assert test_records == source_test
    assert captured == {
        "dataset_id": prepare_data.DATASET_ID,
        "revision": "851f9946252e90c665cdb3cc3eedb78f1f26197c",
        "data_files": {
            "train": "splice_sites_all/train.parquet",
            "test": "splice_sites_all/test.parquet",
        },
        "cache_dir": "/tmp/hf-cache",
    }


def test_source_contract_rejects_changed_counts_sequences_and_labels():
    valid = {
        "train": [
            _record(0, label=0, sequence="A" * 600),
            _record(1, label=1, sequence="C" * 600),
        ],
        "test": [_record(2, label=2, chromosome="chr2", sequence="G" * 600)],
    }
    expected_counts = {"train": 2, "test": 1}

    prepare_data._validate_source_contract(valid, expected_counts=expected_counts)

    with pytest.raises(ValueError, match="expected 2 .* records, observed 1"):
        prepare_data._validate_source_contract(
            {"train": valid["train"][:1], "test": valid["test"]}, expected_counts=expected_counts
        )

    invalid_sequence = {name: [dict(record) for record in records] for name, records in valid.items()}
    invalid_sequence["test"][0]["sequence"] = "G" * 599
    with pytest.raises(ValueError, match="expected 600 bases, observed 599"):
        prepare_data._validate_source_contract(invalid_sequence, expected_counts=expected_counts)

    invalid_label = {name: [dict(record) for record in records] for name, records in valid.items()}
    invalid_label["train"][0]["label"] = True
    with pytest.raises(ValueError, match="requires integer labels"):
        prepare_data._validate_source_contract(invalid_label, expected_counts=expected_counts)

    missing_label = {name: [dict(record) for record in records] for name, records in valid.items()}
    missing_label["test"][0]["label"] = 1
    with pytest.raises(ValueError, match=r"expected \[0, 1, 2\], observed \[0, 1\]"):
        prepare_data._validate_source_contract(missing_label, expected_counts=expected_counts)


def test_stratified_validation_split_is_deterministic_and_leakage_safe():
    records = [_record(index) for index in range(30)]
    records[3] = _record(3, label=0, start=100)

    train_records, validation_records = prepare_data.stratified_train_validation_split(
        records, validation_fraction=0.2, seed=7
    )
    repeated_train, repeated_validation = prepare_data.stratified_train_validation_split(
        records, validation_fraction=0.2, seed=7
    )

    assert len(train_records) == 24
    assert len(validation_records) == 6
    assert Counter(record["label"] for record in validation_records) == {0: 2, 1: 2, 2: 2}
    assert _record_names(train_records) == _record_names(repeated_train)
    assert _record_names(validation_records) == _record_names(repeated_validation)
    assert (
        prepare_data.audit_split_leakage({"train": train_records, "validation": validation_records})["status"]
        == "passed"
    )

    overlapping_names = {records[0]["name"], records[3]["name"]}
    assert overlapping_names <= set(_record_names(train_records)) or overlapping_names <= set(
        _record_names(validation_records)
    )


def test_audit_rejects_exact_sequence_leakage():
    train_record = _record(0, sequence="AACCGG")
    validation_record = _record(1, chromosome="chr2", sequence="aaccgg")

    with pytest.raises(prepare_data.DataLeakageError, match="Exact sequence leakage"):
        prepare_data.audit_split_leakage({"train": [train_record], "validation": [validation_record]})


def test_audit_rejects_genomic_interval_leakage_but_allows_adjacent_windows():
    train_record = _record(0, start=100)
    overlapping_test_record = _record(1, start=699, sequence="TTTT")
    adjacent_test_record = _record(2, start=700, sequence="GGGG")

    with pytest.raises(prepare_data.DataLeakageError, match="Genomic interval leakage"):
        prepare_data.audit_split_leakage({"train": [train_record], "test": [overlapping_test_record]})

    report = prepare_data.audit_split_leakage({"train": [train_record], "test": [adjacent_test_record]})
    assert report["split_pairs"]["train__test"] == {
        "exact_duplicate_sequences": 0,
        "genomic_interval_overlaps": 0,
    }


def test_iid_and_dirichlet_partitions_are_deterministic_and_disjoint():
    records = [_record(index) for index in range(60)]

    iid_sites = prepare_data.partition_records(records, num_sites=4, partition="iid", seed=11)
    repeated_iid_sites = prepare_data.partition_records(records, num_sites=4, partition="iid", seed=11)
    assert {site: _record_names(site_records) for site, site_records in iid_sites.items()} == {
        site: _record_names(site_records) for site, site_records in repeated_iid_sites.items()
    }
    assert sorted(name for site_records in iid_sites.values() for name in _record_names(site_records)) == sorted(
        _record_names(records)
    )
    for label in range(3):
        label_counts = [sum(record["label"] == label for record in site) for site in iid_sites.values()]
        assert max(label_counts) - min(label_counts) <= 1

    dirichlet_sites = prepare_data.partition_records(
        records, num_sites=4, partition="dirichlet", dirichlet_alpha=0.1, seed=11
    )
    repeated_dirichlet_sites = prepare_data.partition_records(
        records, num_sites=4, partition="dirichlet", dirichlet_alpha=0.1, seed=11
    )
    assert {site: _record_names(site_records) for site, site_records in dirichlet_sites.items()} == {
        site: _record_names(site_records) for site, site_records in repeated_dirichlet_sites.items()
    }
    assert sorted(name for site_records in dirichlet_sites.values() for name in _record_names(site_records)) == sorted(
        _record_names(records)
    )
    assert all(dirichlet_sites.values())
    assert any(
        Counter(record["label"] for record in dirichlet_sites[site])
        != Counter(record["label"] for record in iid_sites[site])
        for site in iid_sites
    )


@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_partitions_keep_related_genomic_windows_at_one_site(partition):
    records = [_record(index) for index in range(60)]
    records[3] = _record(3, label=0, start=100)
    records[6] = _record(6, label=0, chromosome="chr2", sequence=records[0]["sequence"])

    sites = prepare_data.partition_records(records, num_sites=3, partition=partition, dirichlet_alpha=0.5, seed=23)

    related_names = {records[index]["name"] for index in (0, 3, 6)}
    assert any(related_names <= set(_record_names(site_records)) for site_records in sites.values())
    assert prepare_data.audit_split_leakage(sites)["status"] == "passed"


def test_prepare_and_write_records_creates_expected_jsonl_and_manifest(tmp_path):
    train_records = [_record(index) for index in range(60)]
    test_records = [_record(index + 100, chromosome="chr2") for index in range(9)]
    prepared = prepare_data.prepare_records(
        train_records,
        test_records,
        num_sites=3,
        partition="dirichlet",
        dirichlet_alpha=0.5,
        validation_fraction=0.1,
        showcase_size=30,
        seed=17,
    )
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "pooled.jsonl").write_text("legacy pooled data\n", encoding="utf-8")
    (train_dir / "site-4.jsonl").write_text("stale site data\n", encoding="utf-8")
    manifest = prepare_data.write_prepared_data(
        tmp_path,
        prepared,
        num_sites=3,
        partition="dirichlet",
        dirichlet_alpha=0.5,
        validation_fraction=0.1,
        showcase_size=30,
        seed=17,
    )

    validation_records = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()]
    saved_test_records = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    site_records = {
        site_name: [json.loads(line) for line in (tmp_path / "train" / f"{site_name}.jsonl").read_text().splitlines()]
        for site_name in ("site-1", "site-2", "site-3")
    }

    assert sum(len(records) for records in site_records.values()) == 30
    assert len(validation_records) == 6
    assert len(saved_test_records) == 9
    assert saved_test_records == test_records
    assert all(
        tuple(record) == prepare_data.OUTPUT_FIELDS
        for records in (validation_records, saved_test_records, *site_records.values())
        for record in records
    )
    assert len(set(_record_names(record for records in site_records.values() for record in records))) == 30
    assert manifest == json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["source"]["revision"] == prepare_data.DATASET_REVISION
    assert manifest["counts"] == {
        "source_train": 60,
        "available_train": 54,
        "train": 30,
        "validation": 6,
        "test": 9,
        "sites": {
            site_name: {
                "count": len(records),
                "label_histogram": {
                    str(label): count for label, count in sorted(Counter(r["label"] for r in records).items())
                },
            }
            for site_name, records in site_records.items()
        },
    }
    assert manifest["audit"]["status"] == "passed"
    assert set(manifest["audit"]["site_pairs"]) == {
        "site-1__site-2",
        "site-1__site-3",
        "site-2__site-3",
    }
    assert all(
        pair_audit == {"exact_duplicate_sequences": 0, "genomic_interval_overlaps": 0}
        for pair_audit in manifest["audit"]["site_pairs"].values()
    )
    assert not (tmp_path / "train" / "pooled.jsonl").exists()
    assert not (tmp_path / "train" / "site-4.jsonl").exists()
    assert "pooled_train" not in manifest["files"]
    assert "pooled_train" not in manifest["file_identities"]
    assert set(manifest["file_identities"]["sites"]) == {"site-1", "site-2", "site-3"}


def test_showcase_sample_is_exact_and_stratified():
    records = [_record(index) for index in range(30)]

    sample = prepare_data.stratified_sample(records, sample_size=9, seed=3)

    assert len(sample) == 9
    assert Counter(record["label"] for record in sample) == {0: 3, 1: 3, 2: 3}
