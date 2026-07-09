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
