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

from nvflare.tool.agent.dataset_inspect import inspect_dataset
from nvflare.tool.agent.inspector import inspect_path

HEADER = "age,occupation,income\n"
ROWS = "39,clerical,100\n50,managerial,220\n38,service,90\n41,clerical,150\n"


def _write_site(root, site, text):
    d = root / site
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.csv").write_text(text, encoding="utf-8")


def test_tabular_with_header_classifies_and_extracts_schema(tmp_path):
    for site in ("site-1", "site-2"):
        _write_site(tmp_path, site, HEADER + ROWS)

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert result["modality"] == "tabular"
    assert result["layout"] == "per_site_directories"
    site = result["sites"][0]
    assert site["header"] == "present"
    assert site["features"] == ["age", "occupation", "income"]
    assert site["dtypes"] == ["numeric", "text", "numeric"]
    assert site["row_count"] == 4
    assert result["schema_agreement"]["status"] == "consistent"
    # metadata only: no cell values anywhere in the block
    assert "clerical" not in str(result)


def test_headerless_tabular_is_ambiguous_with_no_invented_names(tmp_path):
    for site in ("site-1", "site-2"):
        _write_site(tmp_path, site, ROWS)

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["header"] == "ambiguous"
    assert site["features"] is None
    assert site["column_count"] == 3


def test_schema_drift_across_sites_is_reported(tmp_path):
    _write_site(tmp_path, "site-1", HEADER + ROWS)
    _write_site(tmp_path, "site-2", "age,job,income\n" + ROWS)

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    agreement = result["schema_agreement"]
    assert agreement["status"] == "mismatch"
    assert agreement["mismatches"][0]["site"] == "site-2"


def test_image_folder_classifies_by_extension_census(tmp_path):
    for site in ("site-1", "site-2"):
        d = tmp_path / site
        d.mkdir()
        for i in range(3):
            (d / f"scan_{i}.png").write_bytes(b"\x89PNG not really")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert result["modality"] == "image"
    assert result["file_census"] == {".png": 6}
    assert [s["data_files"] for s in result["sites"]] == [3, 3]


def test_directory_without_data_files_returns_none(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    assert inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024) is None


def test_sharded_site_aggregates_rows_and_marks_approximate(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "part-0.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "part-1.csv").write_text(ROWS, encoding="utf-8")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["data_files"] == 2
    # 4 data rows in the schema shard + 4 lines in the second shard
    assert site["row_count"] == 8
    assert site["row_count_approximate"] is True


def test_dataset_scan_accounts_for_reads(tmp_path):
    _write_site(tmp_path, "site-1", HEADER + ROWS)

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert result["scan"]["files_read"] == 1
    assert result["scan"]["bytes_read"] == len((HEADER + ROWS).encode())


def test_non_data_clutter_does_not_exhaust_the_data_file_limit(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(20):
        (d / f"note_{i}.txt").write_text("clutter", encoding="utf-8")
    (d / "zz_data.csv").write_text(HEADER + ROWS, encoding="utf-8")

    result = inspect_dataset(tmp_path, max_files=5, max_file_bytes=512 * 1024)

    assert result is not None
    assert result["modality"] == "tabular"
    assert result["counts_approximate"] is False


def test_parquet_without_reader_marks_schema_unavailable(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("pyarrow unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "data.parquet").write_bytes(b"PAR1 not really")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["format"] == "parquet"
    assert site["schema_available"] is False
    assert site["features"] is None


def test_mixed_modality_is_reported_but_not_routed(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "data.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "scan.png").write_bytes(b"\x89PNG not really")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)
    result = inspect_path(tmp_path)

    assert dataset["modality"] == "mixed"
    assert result["target_type"] == "unknown_target"
    # mixed is the ambiguous case: it routes to orient, never to fed-stats
    # and never to an empty recommendation
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_dtype_drift_with_same_names_is_a_mismatch(tmp_path):
    _write_site(tmp_path, "site-1", HEADER + ROWS)
    # same feature names, but income is text at site-2
    _write_site(tmp_path, "site-2", HEADER + "39,clerical,low\n50,managerial,high\n38,service,low\n41,clerical,mid\n")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    agreement = result["schema_agreement"]
    assert agreement["status"] == "mismatch"
    assert agreement["mismatches"][0]["issue"] == "dtypes_differ"


def test_directory_only_tree_is_bounded(tmp_path, monkeypatch):
    import nvflare.tool.agent.dataset_inspect as di

    monkeypatch.setattr(di, "MAX_WALK_ENTRIES", 10)
    d = tmp_path
    for i in range(30):
        d = d / f"level_{i}"
        d.mkdir()
    (d / "data.csv").write_text(HEADER + ROWS, encoding="utf-8")

    groups, census, truncated = di._collect_data_files(tmp_path, max_files=250)

    assert truncated is True  # the entries cap stopped the walk instead of running the full depth


def test_feature_names_are_sanitized_and_capped(tmp_path):
    long_name = "x" * 500
    header = f'age,"bad\x07name",{long_name}\n'
    _write_site(tmp_path, "site-1", header + "39,a,1\n50,b,2\n38,c,3\n")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["header"] == "present"
    assert site["features"][1] == "badname"  # control character stripped
    assert len(site["features"][2]) == 120
    assert site["feature_names_truncated"] is True


def test_degenerate_file_keeps_stable_site_shape(tmp_path):
    _write_site(tmp_path, "site-1", "just_one_line\n")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    for key in ("format", "header", "column_count", "features", "dtypes", "row_count", "row_count_approximate"):
        assert key in site
        assert site[key] is None or key == "format"


def test_nan_tokens_are_not_numeric_evidence(tmp_path):
    _write_site(tmp_path, "site-1", "age,flag\n39,nan\n50,nan\n38,nan\n")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert result["sites"][0]["dtypes"] == ["numeric", "text"]


def test_parquet_sharded_site_sums_exact_rows(tmp_path):
    pa = __import__("pytest").importorskip("pyarrow")
    import pyarrow.parquet as pq

    d = tmp_path / "site-1"
    d.mkdir()
    table1 = pa.table({"age": [39, 50], "income": ["a", "b"]})
    table2 = pa.table({"age": [38, 41, 44], "income": ["c", "d", "e"]})
    pq.write_table(table1, d / "part-0.parquet")
    pq.write_table(table2, d / "part-1.parquet")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["schema_available"] is True
    assert site["features"] == ["age", "income"]
    assert site["dtypes"] == ["numeric", "text"]
    assert site["row_count"] == 5
    assert site["row_count_approximate"] is False
    assert result["scan"]["files_read"] == 2


def test_more_tabular_than_stray_images_is_tabular(tmp_path):
    # 2 CSVs + 1 stray PNG in one site: within both tolerance windows, the
    # substantive side (more files) wins
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "a.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "b.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "scan.png").write_bytes(b"\x89PNG not really")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)
    result = inspect_path(tmp_path)

    assert dataset["modality"] == "tabular"
    assert result["target_type"] == "tabular_dataset"
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-fed-stats"]


def test_tabular_shards_with_one_stray_plot_stay_tabular(tmp_path):
    # the companion-cap regression case: 3 CSV shards + 1 exported plot must
    # not flip to mixed because the companion window widened to 4
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(3):
        (d / f"part_{i}.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "plot.png").write_bytes(b"\x89PNG not really")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert dataset["modality"] == "tabular"


def test_stray_image_below_threshold_stays_tabular(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(20):
        (d / f"part_{i}.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "plot.png").write_bytes(b"\x89PNG not really")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert dataset["modality"] == "tabular"
    assert dataset["file_census"][".png"] == 1  # stray file stays visible


def test_image_only_site_among_tabular_sites_is_mixed_and_unrouted(tmp_path):
    # Codex repro: 19 tabular sites + 1 image-only site cannot be served by
    # one tabular job; the dataset is mixed and stays unrouted.
    for i in range(1, 20):
        _write_site(tmp_path, f"site-{i:02d}", HEADER + ROWS)
    d = tmp_path / "site-99"
    d.mkdir()
    (d / "scan.png").write_bytes(b"\x89PNG not really")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)
    result = inspect_path(tmp_path)

    assert dataset["modality"] == "mixed"
    assert result["target_type"] == "unknown_target"
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    image_site = [s for s in dataset["sites"] if s["name"] == "site-99"][0]
    assert image_site["image_files"] == 1 and image_site["tabular_files"] == 0


def test_image_dataset_with_companion_labels_stays_image(tmp_path):
    # THE standard imaging shape: scans plus a labels.csv per site.
    for site in ("site-1", "site-2"):
        d = tmp_path / site
        d.mkdir()
        for i in range(12):
            (d / f"scan_{i}.png").write_bytes(b"\x89PNG not really")
        (d / "labels.csv").write_text(HEADER + ROWS, encoding="utf-8")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert dataset["modality"] == "image"
    assert dataset["sites"][0]["tabular_companions"] == 1


def test_three_label_files_per_site_stays_image(tmp_path):
    # train/val/test label files are common companion metadata
    for site in ("site-1", "site-2"):
        d = tmp_path / site
        d.mkdir()
        for i in range(12):
            (d / f"scan_{i}.png").write_bytes(b"\x89PNG not really")
        for split in ("train", "val", "test"):
            (d / f"labels_{split}.csv").write_text(HEADER + ROWS, encoding="utf-8")

    dataset = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert dataset["modality"] == "image"
    assert dataset["sites"][0]["tabular_companions"] == 3


def test_same_width_shard_header_drift_is_flagged(tmp_path):
    # Codex repro: part-1 header renames occupation to job at the same width.
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "part-0.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "part-1.csv").write_text("age,job,income\n" + ROWS, encoding="utf-8")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert result["sites"][0]["shard_schema_consistent"] is False
    assert result["schema_agreement"]["status"] == "mismatch"


def test_repeated_identical_shard_header_is_consistent(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "part-0.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "part-1.csv").write_text(HEADER + ROWS, encoding="utf-8")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert "shard_schema_consistent" not in site  # consistent
    assert site["row_count"] == 8  # detected repeated header is not a data row
    assert site["row_count_approximate"] is True  # multi-file totals stay approximate


def test_headerless_data_shard_is_consistent(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "part-0.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "part-1.csv").write_text(ROWS, encoding="utf-8")  # data-first shard

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    assert "shard_schema_consistent" not in result["sites"][0]


def test_parquet_shard_schema_disagreement_is_a_mismatch(tmp_path):
    pa = __import__("pytest").importorskip("pyarrow")
    import pyarrow.parquet as pq

    d = tmp_path / "site-1"
    d.mkdir()
    pq.write_table(pa.table({"age": [39, 50]}), d / "part-0.parquet")
    pq.write_table(pa.table({"height": [170, 180, 175]}), d / "part-1.parquet")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["shard_schema_consistent"] is False
    assert site["row_count_approximate"] is True
    agreement = result["schema_agreement"]
    assert agreement["status"] == "mismatch"
    assert {"site": "site-1", "reference_site": "site-1", "issue": "shards_differ"} in agreement["mismatches"]


def test_csv_shard_column_count_drift_is_flagged(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "part-0.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "part-1.csv").write_text("39,100\n50,220\n", encoding="utf-8")  # 2 columns, not 3

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["shard_schema_consistent"] is False
    assert result["schema_agreement"]["status"] == "mismatch"


def test_mixed_format_site_is_never_exact(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "data.csv").write_text(HEADER + ROWS, encoding="utf-8")
    (d / "extra.parquet").write_bytes(b"PAR1 not really")

    result = inspect_dataset(tmp_path, max_files=250, max_file_bytes=512 * 1024)

    site = result["sites"][0]
    assert site["row_count_approximate"] is True
    assert site["shard_schema_consistent"] is False


def test_file_limit_marks_counts_approximate(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(10):
        (d / f"img_{i}.png").write_bytes(b"x")

    result = inspect_dataset(tmp_path, max_files=5, max_file_bytes=512 * 1024)

    assert result["counts_approximate"] is True


def test_inspect_path_routes_tabular_dataset_to_fed_stats(tmp_path):
    for site in ("site-1", "site-2"):
        _write_site(tmp_path, site, HEADER + ROWS)

    result = inspect_path(tmp_path)

    assert result["target_type"] == "tabular_dataset"
    assert result["dataset"]["modality"] == "tabular"
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-fed-stats"]


def test_inspect_path_routes_image_dataset_to_fed_stats(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    (d / "scan.png").write_bytes(b"\x89PNG not really")

    result = inspect_path(tmp_path)

    assert result["target_type"] == "image_dataset"
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-fed-stats"]


def test_truncated_dataset_recommends_only_fed_stats(tmp_path, monkeypatch):
    # A classified dataset keeps a single recommendation even when the walk
    # truncated (classification_incomplete would otherwise add orient).
    import nvflare.tool.agent.inspector as inspector_module

    monkeypatch.setattr(inspector_module, "DEFAULT_MAX_FILES", 3)
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(10):
        (d / f"part_{i}.csv").write_text(HEADER + ROWS, encoding="utf-8")

    result = inspect_path(tmp_path, max_files=3)

    assert result["target_type"] == "tabular_dataset"
    assert result["skill_selection"]["recommended_skills"] == ["nvflare-fed-stats"]


def test_inspect_path_max_files_flows_to_dataset_walk(tmp_path):
    d = tmp_path / "site-1"
    d.mkdir()
    for i in range(10):
        (d / f"part_{i}.csv").write_text(HEADER + ROWS, encoding="utf-8")

    capped = inspect_path(tmp_path, max_files=3)
    uncapped = inspect_path(tmp_path, max_files=250)

    assert capped["dataset"]["counts_approximate"] is True
    assert uncapped["dataset"]["counts_approximate"] is False
    assert uncapped["dataset"]["sites"][0]["data_files"] == 10


def test_inspect_path_keeps_code_classification_priority(tmp_path):
    # A training repo that also contains CSVs stays a code target: dataset
    # classification only runs when code classification found nothing.
    (tmp_path / "train.py").write_text(
        "import torch\nfrom torch.optim import Adam\n\n\ndef main():\n    pass\n", encoding="utf-8"
    )
    _write_site(tmp_path, "data", HEADER + ROWS)

    result = inspect_path(tmp_path)

    assert result["target_type"] != "tabular_dataset"
    assert result["dataset"] is None
    assert "nvflare-fed-stats" not in result["skill_selection"]["recommended_skills"]
