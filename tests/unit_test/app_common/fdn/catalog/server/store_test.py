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

import json

import pytest

from nvflare.app_common.fdn.catalog.common.payload import ResultStatus, Selection, build_result
from nvflare.app_common.fdn.catalog.common.schema import Lineage
from nvflare.app_common.fdn.catalog.server.reconcile import Reconciler
from nvflare.app_common.fdn.catalog.server.search import SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore, manifest_key
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog

NOW = "2026-06-01T00:00:00Z"
STUDY, CLIENT = "study1", "site-a"


def seeded(tmp_path):
    store = CatalogStore(str(tmp_path))
    result = build_result(ResultStatus.OK, [make_catalog()])
    Reconciler(store, LIMITS).apply(STUDY, CLIENT, Selection(None), result, NOW)
    return store


class TestPersistence:
    def test_state_survives_a_restart(self, tmp_path):
        before = seeded(tmp_path)
        after = CatalogStore(str(tmp_path))
        assert after.client_state(STUDY, CLIENT) == before.client_state(STUDY, CLIENT)
        assert after.search(STUDY, SearchQuery()).total == 1

    def test_a_failed_file_replace_changes_neither_the_disk_nor_the_memory_copy(self, tmp_path, monkeypatch):
        import nvflare.app_common.fdn.catalog.server.store as store_module

        store = seeded(tmp_path)
        before = store.client_state(STUDY, CLIENT)
        changed = store.client_state(STUDY, CLIENT)
        changed["client"]["health"] = "FAILED"

        def broken(path, data):
            raise OSError("disk full")

        monkeypatch.setattr(store_module, "write_atomically", broken)
        with pytest.raises(OSError):
            store.commit(STUDY, CLIENT, changed)
        assert store.client_state(STUDY, CLIENT) == before
        assert CatalogStore(str(tmp_path)).client_state(STUDY, CLIENT) == before

    def test_corrupt_client_file_is_unknown_and_isolated(self, tmp_path, caplog):
        seeded(tmp_path)
        (tmp_path / "fdn" / "catalog" / STUDY / "site-b.json").write_text("{not json")
        store = CatalogStore(str(tmp_path))
        assert "unreadable and starts empty: site-b" in caplog.text and CLIENT not in caplog.text
        assert store.client_state(STUDY, "site-b")["client"]["health"] == "UNKNOWN"
        assert store.search(STUDY, SearchQuery()).total == 1

    def test_a_corrupt_client_file_is_kept_aside_and_not_overwritten_by_the_next_commit(self, tmp_path, caplog):
        seeded(tmp_path)
        path = tmp_path / "fdn" / "catalog" / STUDY / "site-b.json"
        path.write_text("{not json")
        store = CatalogStore(str(tmp_path))
        kept = path.with_name("site-b.json.damaged")
        assert kept.read_text() == "{not json" and not path.exists()
        assert "site-b.json.damaged" in caplog.text
        store.commit(STUDY, "site-b", store.client_state(STUDY, "site-b"))
        assert kept.read_text() == "{not json"  # a later commit replaces the live file only
        assert CatalogStore(str(tmp_path)).client_state(STUDY, "site-b")["client"]["health"] == "UNKNOWN"

    @pytest.mark.parametrize(
        "damage",
        [
            lambda s: s["client"].pop("failures"),
            lambda s: s.update(datasets=[]),
            lambda s: s["datasets"].update(d={"available": True}),
            lambda s: s["datasets"]["clinical-research"].update(metadata=None),
            lambda s: s["datasets"]["clinical-research"].update(metadata={"tables": None}),
            lambda s: s["datasets"]["clinical-research"].update(catalog_version="1"),
            lambda s: s["datasets"]["clinical-research"].update(available=1),
            lambda s: s["client"].update(failures=None),
            lambda s: s["client"].update(unknown_field=1),
        ],
    )
    def test_structurally_invalid_state_is_rebuilt_for_that_client_only(self, tmp_path, damage):
        store = seeded(tmp_path)
        store.commit(STUDY, "site-b", store.client_state(STUDY, "site-b"))
        state = json.loads((tmp_path / "fdn" / "catalog" / STUDY / "site-a.json").read_text())
        damage(state)
        (tmp_path / "fdn" / "catalog" / STUDY / "site-a.json").write_text(json.dumps(state))
        reloaded = CatalogStore(str(tmp_path))
        assert reloaded.client_state(STUDY, "site-b")["client"]["health"] == "UNKNOWN"  # never had a result
        assert reloaded.client_state(STUDY, "site-a")["datasets"] == {}
        reloaded.search(STUDY, SearchQuery())  # the damaged file never reaches a search

    def test_unknown_format_is_refused_not_guessed(self, tmp_path):
        seeded(tmp_path)
        path = tmp_path / "fdn" / "catalog" / STUDY / f"{CLIENT}.json"
        data = json.loads(path.read_text())
        data["format"] = 99
        path.write_text(json.dumps(data))
        assert CatalogStore(str(tmp_path)).client_state(STUDY, CLIENT)["datasets"] == {}

    def test_leftover_temp_file_is_removed(self, tmp_path):
        seeded(tmp_path)
        tmp = tmp_path / "fdn" / "catalog" / STUDY / "site-a.json.tmp"
        tmp.write_text("partial")
        CatalogStore(str(tmp_path))
        assert not tmp.exists()

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
    def test_names_cannot_escape_the_store(self, tmp_path, bad):
        with pytest.raises(ValueError):
            CatalogStore(str(tmp_path)).client_state(STUDY, bad)
        with pytest.raises(ValueError):
            CatalogStore(str(tmp_path)).client_state(bad, CLIENT)


class TestManifestKey:
    def test_the_key_is_created_once_owner_only_and_then_kept(self, tmp_path):
        key = manifest_key(str(tmp_path))
        path = tmp_path / "fdn" / "manifest.key"
        assert len(key) == 32 and manifest_key(str(tmp_path)) == key and path.stat().st_mode & 0o077 == 0

    @pytest.mark.parametrize("damage", [b"", b"short"])
    def test_a_key_file_left_empty_or_short_by_a_crash_is_replaced_not_trusted(self, tmp_path, damage):
        path = tmp_path / "fdn" / "manifest.key"
        path.parent.mkdir(parents=True)
        path.write_bytes(damage)
        key = manifest_key(str(tmp_path))
        assert len(key) == 32 and path.read_bytes() == key


class TestFilePermissions:
    def test_stored_catalogs_are_readable_by_the_owner_only(self, tmp_path):
        seeded(tmp_path)
        path = tmp_path / "fdn" / "catalog" / STUDY / f"{CLIENT}.json"
        assert path.stat().st_mode & 0o077 == 0


class TestIndexed:
    def test_the_names_held_for_each_client_of_a_study(self, tmp_path):
        store = seeded(tmp_path)
        store.commit(STUDY, "site-b", store.client_state(STUDY, "site-b"))
        assert store.indexed(STUDY) == {CLIENT: ["clinical-research"], "site-b": []}
        assert store.indexed("other") == {}


class TestHealth:
    def test_health_report(self, tmp_path):
        store = seeded(tmp_path)
        health = store.health(STUDY)
        assert health[0]["client"] == CLIENT and health[0]["health"] == "OK"
        assert [h["client"] for h in health] == [CLIENT]


class TestSearch:
    @pytest.fixture
    def store(self, tmp_path):
        store = CatalogStore(str(tmp_path))
        rec = Reconciler(store, LIMITS)
        for i, client in enumerate(["site-a", "site-b", "site-c"]):
            catalogs = [
                make_catalog(dataset="omop", display_name=f"OMOP {client}", modalities=("clinical_tabular",)),
                make_catalog(dataset="fhir", display_name="FHIR bundle", modalities=("clinical_text",)),
            ]
            rec.apply(STUDY, client, Selection(None), build_result(ResultStatus.OK, catalogs), NOW)
        return store

    def test_lineage_never_names_a_withheld_dataset(self, store):
        rec = Reconciler(store, LIMITS)
        derived = make_catalog(dataset="derived", lineage=(Lineage("omop", "v1"), Lineage("fhir", "v2")))
        rec.apply(
            STUDY,
            "site-a",
            Selection(None),
            build_result(ResultStatus.OK, [derived, make_catalog(dataset="fhir")], withheld=["omop"]),
            NOW,
        )
        entry = store.search(STUDY, SearchQuery(client="site-a", dataset="derived")).entries[0]
        assert [x.dataset for x in entry.catalog.lineage] == ["fhir"]

    def test_filters(self, store):
        assert store.search(STUDY, SearchQuery(client="site-b")).total == 2
        assert store.search(STUDY, SearchQuery(text="omop")).total == 3
        assert store.search(STUDY, SearchQuery(modality="clinical_text")).total == 3
        assert (
            store.search(STUDY, SearchQuery(standard="OMOP", standard_version="5.4", operation="cohort.count")).total
            == 6
        )
        assert store.search(STUDY, SearchQuery(standard="FHIR")).total == 0

    def test_paging_is_stable_and_complete(self, store):
        seen, cursor = [], None
        for _ in range(2):
            page = store.search(STUDY, SearchQuery(page_size=4, cursor=cursor))
            seen += [(e.client, e.dataset) for e in page.entries]
            cursor = page.next_cursor
            if cursor is None:
                break
        else:
            pytest.fail("pagination did not finish after the two expected pages")
        assert seen == sorted(seen) and len(seen) == 6 and page.total == 6

    def test_cursor_is_bound_to_the_filters_and_study(self, store):
        cursor = store.search(STUDY, SearchQuery(page_size=2)).next_cursor
        with pytest.raises(ValueError):
            store.search(STUDY, SearchQuery(page_size=2, cursor=cursor, client="site-a"))
        with pytest.raises(ValueError):
            store.search("other", SearchQuery(page_size=2, cursor=cursor))
        with pytest.raises(ValueError):
            store.search(STUDY, SearchQuery(cursor="garbage"))

    def test_page_size_bounds(self, store):
        for size in (0, 101):
            with pytest.raises(ValueError):
                store.search(STUDY, SearchQuery(page_size=size))

    def test_facets_cover_the_filtered_set(self, store):
        facets = store.search(STUDY, SearchQuery(client="site-a")).facets
        assert facets["modality"] == {"clinical_tabular": 1, "clinical_text": 1}
        assert facets["standard"] == {"OMOP": 2}

    def test_unavailable_entry_exposes_no_metadata_and_matches_no_metadata_filter(self, store):
        rec = Reconciler(store, LIMITS)
        rec.apply(STUDY, "site-a", Selection(None), build_result(ResultStatus.OK, unavailable=["omop"]), NOW)
        entry = next(e for e in store.search(STUDY, SearchQuery(client="site-a", dataset="omop")).entries)
        assert entry.available is False and entry.catalog is None and entry.catalog_version is None
        assert store.search(STUDY, SearchQuery(client="site-a", text="omop")).total == 0
        assert store.search(STUDY, SearchQuery(available=False)).total == 1


def test_search_reuses_client_views_and_invalidates_withholding_on_commit(tmp_path, monkeypatch):
    store = seeded(tmp_path)
    original = store._client_entries
    rebuilt = []

    def build(client, state):
        assert not store.lock._is_owned()
        rebuilt.append(client)
        return original(client, state)

    monkeypatch.setattr(store, "_client_entries", build)
    assert store.search(STUDY, SearchQuery()).total == 1
    assert store.search(STUDY, SearchQuery(page_size=1)).total == 1
    assert rebuilt == [CLIENT]
    changed = store.client_state(STUDY, CLIENT)
    changed["datasets"]["clinical-research"]["withheld"] = True
    store.commit(STUDY, CLIENT, changed)
    assert store.search(STUDY, SearchQuery()).total == 0
    assert rebuilt == [CLIENT, CLIENT]
