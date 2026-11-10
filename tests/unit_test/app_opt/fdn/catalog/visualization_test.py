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

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from nvflare.app_common.fdn.catalog.common.payload import SiteEntry, SiteStatus, build_catalog_results
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ColumnInfo, Provenance, Relationship, TableInfo
from nvflare.app_opt.fdn.catalog.visualization import CatalogVisualization


def catalogs():
    person = TableInfo(
        "person", (ColumnInfo("id", "INTEGER", False), ColumnInfo("age", "INTEGER", None)), primary_key=("id",)
    )
    visit = TableInfo(
        "visit",
        (ColumnInfo("person_id", "INTEGER", False),),
        relationships=(Relationship(("person_id",), "person", ("id",)),),
    )
    first = CatalogInfo("omop", "OMOP", Provenance("parquet"), tables=(person, visit, TableInfo("empty", ())))
    second = CatalogInfo(
        "omop",
        "OMOP",
        Provenance("sql"),
        tables=(replace(person, columns=(ColumnInfo("id", "BIGINT", False), ColumnInfo("consent", "TEXT", True))),),
    )
    return first, second


def make_results(extra_site_1=()):
    """A catalog read: site-1 and site-2 publish omop, site-3 reports it unavailable, site-4 gives no result."""
    first, second = catalogs()
    return build_catalog_results(
        "default",
        "job-1",
        "2026-11-09T00:00:00Z",
        {
            "site-1": SiteEntry(SiteStatus.OK, None, (first, *extra_site_1)),
            "site-2": SiteEntry(SiteStatus.OK, None, (second,), (), ("big",)),
            "site-3": SiteEntry(SiteStatus.OK, None, (), ("omop",)),
            "site-4": SiteEntry(SiteStatus.NO_RESULT),
        },
    )


@pytest.fixture
def results():
    return make_results()


@pytest.mark.parametrize("source", ["dict", "file", "path_str"])
def test_loads_a_catalog_read_result_without_modifying_it(results, tmp_path, source):
    original = copy.deepcopy(results)
    if source == "dict":
        data = results
    else:
        data = tmp_path / "catalog_results.json"
        data.write_text(json.dumps(results))
        data = str(data) if source == "path_str" else data
    view = CatalogVisualization(data)
    frame = view.get_datasets()
    assert list(zip(frame["site"], frame["dataset"], frame["status"])) == [
        ("site-1", "omop", "published"),
        ("site-2", "big", "skipped"),
        ("site-2", "omop", "published"),
        ("site-3", "omop", "unavailable"),
    ]
    assert frame["source"].tolist() == ["parquet", None, "sql", None]
    assert pd.isna(frame.loc[3, "tables"])  # unavailable never means an empty database
    assert results == original


def test_site_status_shows_every_targeted_site(results):
    sites = CatalogVisualization(results).get_sites()
    assert sites.to_dict("records") == [
        {"site": "site-1", "status": "OK", "reason_code": None, "published": 1, "unavailable": 0, "skipped": 0},
        {"site": "site-2", "status": "OK", "reason_code": None, "published": 1, "unavailable": 0, "skipped": 1},
        {"site": "site-3", "status": "OK", "reason_code": None, "published": 0, "unavailable": 1, "skipped": 0},
        {"site": "site-4", "status": "NO_RESULT", "reason_code": None, "published": 0, "unavailable": 0, "skipped": 0},
    ]


def mock_styler(monkeypatch):
    """Replace pandas rendering, so a unit test launches no renderer or font-discovery process."""
    styled = Mock()
    styled.format.return_value = styled
    styled.format_index.return_value = styled
    styled.apply.return_value = styled
    monkeypatch.setattr(pd.DataFrame, "style", property(lambda frame: styled))
    return styled


def test_an_unknown_observation_is_shown_as_unknown_not_unpublished(results, monkeypatch):
    styled = mock_styler(monkeypatch)
    CatalogVisualization(results).show_datasets()
    assert styled.format.call_args_list[-1].kwargs == {"escape": "html", "na_rep": "unknown", "subset": ["observed_at"]}


def test_every_view_names_the_study(results, monkeypatch):
    styled = mock_styler(monkeypatch)
    view = CatalogVisualization({**results, "study": "oncology"})
    assert view.study == "oncology"
    for show in (
        view.show_sites,
        view.show_datasets,
        lambda: view.show_tables(dataset="omop"),
        lambda: view.show_schema("person"),
        view.show_table_comparison,
        lambda: view.show_column_comparison("person"),
        view.show_relationships,
    ):
        show()
    captions = [call.args[0] for call in styled.set_caption.call_args_list]
    assert len(captions) == 7 and all("Study oncology</strong>" in caption for caption in captions)


def test_a_result_without_a_study_is_rejected(results):
    with pytest.raises(ValueError, match="names no Study"):
        CatalogVisualization({**results, "study": ""})


def test_tables_schema_and_relationships_preserve_source_metadata(results):
    view = CatalogVisualization(results)
    tables = view.get_tables(site="site-1", dataset="omop")
    assert tables["table"].tolist() == ["person", "visit", "empty"]
    assert tables["columns"].tolist() == [2, 1, 0]
    schema = view.get_schema("person", site="site-1", dataset="omop")
    assert schema["column"].tolist() == ["id", "age"]
    assert schema["position"].tolist() == [1, 2]
    assert schema["primary_key"].tolist() == [True, False]
    assert schema.loc[1, "nullable"] is None
    assert view.get_relationships().to_dict("records") == [
        {"site": "site-1", "dataset": "omop", "table": "visit", "columns": "person_id", "references": "person(id)"}
    ]


def test_table_comparison_distinguishes_empty_tables_missing_tables_and_datasets_not_read(results):
    view = CatalogVisualization(results)
    frame = view.get_table_comparison(dataset="omop")
    assert frame.at["empty", ("site-1", "omop")] == 0
    assert pd.isna(frame.at["visit", ("site-2", "omop")])
    assert frame[("site-3", "omop")].isna().all()
    assert frame.attrs["unavailable"] == [("site-3", "omop")]
    styles = view._comparison_styles(frame, differences=False)
    assert "#fce4e4" in styles.at["visit", ("site-2", "omop")]
    assert "#e5e7eb" in styles.at["visit", ("site-3", "omop")]
    assert styles.at["empty", ("site-1", "omop")] == ""


def test_column_comparison_retains_shared_and_different_columns_and_observed_types(results):
    view = CatalogVisualization(results)
    frame = view.get_column_comparison("person", dataset="omop")
    assert frame.index.tolist() == ["id", "age", "consent"]
    assert frame.at["id", ("site-1", "omop")] == "INTEGER"
    assert frame.at["id", ("site-2", "omop")] == "BIGINT"
    assert pd.isna(frame.at["consent", ("site-1", "omop")])
    styles = view._comparison_styles(frame, differences=True)
    assert "#fff0c2" in styles.at["id", ("site-1", "omop")]
    assert "#fff0c2" in styles.at["id", ("site-2", "omop")]
    assert styles.at["age", ("site-1", "omop")] == ""
    assert "#fce4e4" in styles.at["age", ("site-2", "omop")]
    assert "#e5e7eb" in styles.at["age", ("site-3", "omop")]


def test_nullability_and_keys_are_distinct_from_publication_and_unknown_metadata(results):
    view = CatalogVisualization(results)
    nullable = view.get_column_comparison("person", field="nullable")
    assert nullable.at["age", ("site-1", "omop")] == "unknown"
    assert nullable.at["id", ("site-2", "omop")] == "no"
    assert nullable.at["consent", ("site-2", "omop")] == "yes"
    keys = view.get_column_comparison("person", field="primary_key")
    assert keys.at["id", ("site-1", "omop")] == "yes"
    assert keys.at["age", ("site-1", "omop")] == "no"
    assert pd.isna(keys.at["age", ("site-2", "omop")])
    with pytest.raises(ValueError, match="field must be"):
        view.get_column_comparison("person", field="unrecognized")


def test_multiple_datasets_at_one_site_are_compared_separately():
    first, _ = catalogs()
    view = CatalogVisualization(make_results(extra_site_1=(replace(first, dataset="cohort"),)))
    frame = view.get_column_comparison("person")
    assert frame.columns.tolist()[:3] == [("site-1", "cohort"), ("site-1", "omop"), ("site-2", "big")]
    assert view.get_datasets(site="site-1")["dataset"].tolist() == ["cohort", "omop"]
    assert view.get_table_comparison(dataset="cohort").columns.tolist() == [("site-1", "cohort")]


def test_empty_results_and_selections_have_stable_frames():
    view = CatalogVisualization(build_catalog_results("default", "job-1", "2026-11-09T00:00:00Z", {}))
    assert view.get_sites().empty
    assert view.get_datasets().empty
    assert view.get_tables().empty
    assert view.get_schema("person").empty
    assert view.get_relationships().empty
    assert view.get_table_comparison().empty
    assert view.get_column_comparison("person").empty


def test_rejects_other_documents_invalid_metadata_and_duplicates(results):
    with pytest.raises(ValueError, match="Not a catalog read result"):
        CatalogVisualization({"entries": [], "total": 0})
    with pytest.raises(ValueError, match="file path or its loaded dict"):
        CatalogVisualization([results])
    broken = copy.deepcopy(results)
    broken["datasets"][0]["metadata"]["extra"] = 1
    with pytest.raises(ValueError, match="Invalid catalog metadata for site-1/omop"):
        CatalogVisualization(broken)
    duplicated = copy.deepcopy(results)
    duplicated["sites"]["site-1"]["unavailable"] = ["omop"]
    with pytest.raises(ValueError, match="Duplicate site/dataset"):
        CatalogVisualization(duplicated)


def test_notebook_html_escapes_values_labels_and_captions(monkeypatch):
    styled = Mock()
    styled.format.return_value = styled
    styled.format_index.return_value = styled
    # Control rendering so this unit test cannot launch optional matplotlib/font tools.
    monkeypatch.setattr(pd.DataFrame, "style", property(lambda frame: styled))
    frame = pd.DataFrame([["<script>value</script>"]], index=["<b>row</b>"], columns=["<b>column</b>"])
    assert CatalogVisualization._style(frame, "<script>title</script>", "Compare <b>columns</b> & sources.") is styled
    assert styled.format.call_args.kwargs["escape"] == "html"
    assert [call.kwargs for call in styled.format_index.call_args_list] == [
        {"escape": "html", "axis": 0},
        {"escape": "html", "axis": 1},
    ]
    styled.set_caption.assert_called_once_with(
        "<strong>&lt;script&gt;title&lt;/script&gt;</strong><br>Compare &lt;b&gt;columns&lt;/b&gt; &amp; sources."
    )


@pytest.mark.parametrize("example", ["federated-catalog", "federated-catalog-services"])
def test_offline_notebook_cells_use_result_json_and_reusable_views(results, tmp_path, monkeypatch, example):
    import IPython.display

    result = tmp_path / "catalog_results.json"
    result.write_text(json.dumps(results))
    notebook = json.loads(
        (Path(__file__).resolve().parents[5] / "examples/advanced" / example / "catalog_explorer.ipynb").read_text()
    )
    styled = Mock()
    styled.format.return_value = styled
    styled.format_index.return_value = styled
    styled.apply.return_value = styled
    monkeypatch.setattr(pd.DataFrame, "style", property(lambda frame: styled))
    display = Mock()
    monkeypatch.setattr(IPython.display, "display", display)
    # The notebook receives this controlled result file. Rendering is mocked; no shell,
    # kernel, network, live session or optional font process is started by this unit test.
    namespace = {"CATALOG_FILE": str(result)}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert namespace["vis"].get_datasets()["site"].tolist() == ["site-1", "site-2", "site-2", "site-3"]
    assert styled.apply.called
    assert display.call_count >= 8  # explicit display also works when %run suppresses expression output
    captions = [call.args[0] for call in styled.set_caption.call_args_list]
    assert len(captions) == display.call_count  # every rendered group carries its own title and explanation
    assert all(caption.startswith("<strong>") and caption.partition("</strong><br>")[2] for caption in captions)
    assert any("Column nullability comparison: person" in caption for caption in captions)
    assert not any(name in namespace for name in ("session", "subprocess", "catalog_run", "catalog_json"))


@pytest.mark.parametrize("example", ["federated-catalog", "federated-catalog-services"])
def test_runner_notebook_reads_the_catalog_and_passes_the_result_to_the_explorer(
    results, tmp_path, monkeypatch, example
):
    from types import SimpleNamespace

    import IPython.display
    from IPython.core.inputtransformer2 import TransformerManager

    folder = Path(__file__).resolve().parents[5] / "examples/advanced" / example
    notebook = json.loads((folder / "get_catalog.ipynb").read_text())
    explorer = json.loads((folder / "catalog_explorer.ipynb").read_text())
    styled = Mock()
    styled.format.return_value = styled
    styled.format_index.return_value = styled
    styled.apply.return_value = styled
    monkeypatch.setattr(pd.DataFrame, "style", property(lambda frame: styled))
    display = Mock()
    monkeypatch.setattr(IPython.display, "display", display)
    commands = []
    namespace = {}

    def bash(name, options, body):
        # The external command boundary is controlled. This unit test launches no shell,
        # kernel or federation connection; the fixture stands in for the read's saved result.
        assert name == "bash"
        commands.append(body)

    def run(name, args):
        assert name == "run" and args == "-i catalog_explorer.ipynb"
        # The notebook names the result file job.py saved; redirect it to the fixture written in its place.
        assert namespace["CATALOG_FILE"].endswith("catalog_results.json")
        saved = tmp_path / "catalog_results.json"
        saved.write_text(json.dumps(results))
        namespace["CATALOG_FILE"] = str(saved)
        for cell in explorer["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)

    namespace["get_ipython"] = lambda: SimpleNamespace(
        system=lambda command: commands.append(command), run_cell_magic=bash, run_line_magic=run
    )
    transformer = TransformerManager()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(transformer.transform_cell("".join(cell["source"])), namespace)
    assert any("python job.py" in command for command in commands)
    assert not any("nvflare catalog" in command for command in commands)
    assert namespace["vis"].get_datasets()["source"].tolist() == ["parquet", None, "sql", None]
    assert display.call_count >= 8
