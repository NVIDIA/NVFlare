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
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import job
import pytest
from prepare_data import prepare_data

from nvflare.collab.api.group_call_context import FrozenResultQueue


@pytest.mark.parametrize("empty_sites", [[], ["site-2"], job.SITES])
def test_distributed_matches_centralized(tmp_path, monkeypatch, empty_sites):
    prepare_data(tmp_path)
    partials = []
    with closing(sqlite3.connect(":memory:")) as central:
        central.execute("CREATE TABLE sales (id, region, amount_cents)")
        for site in job.SITES:
            path = tmp_path / f"{site}.db"
            with closing(sqlite3.connect(path)) as db, db:
                if site in empty_sites:
                    db.execute("DELETE FROM sales")
                central.executemany("INSERT INTO sales VALUES (?, ?, ?)", db.execute("SELECT * FROM sales"))
            monkeypatch.setattr(job, "collab", SimpleNamespace(get_app_prop=lambda name: str(path)))
            partials.append((site, job.query()))
        expected = central.execute(
            "SELECT region, COUNT(*), SUM(amount_cents), AVG(amount_cents) FROM sales GROUP BY region ORDER BY region"
        ).fetchall()
    responses = FrozenResultQueue(partials, {}, len(partials))
    group = Mock()
    group.query.return_value = responses
    monkeypatch.setattr(
        job,
        "collab",
        SimpleNamespace(
            get_clients=Mock(return_value=group),
            workspace=SimpleNamespace(get_run_dir=lambda job_id: str(tmp_path)),
            fl_ctx=SimpleNamespace(get_job_id=lambda: "test-job"),
        ),
    )
    actual = job.run()
    assert actual["rows"] == expected
    assert actual["columns"] == ["region", "sale_count", "total_cents", "average_cents"]
    assert json.loads((tmp_path / "result.json").read_text())["rows"] == [list(row) for row in expected]
    job.collab.get_clients.assert_called_once_with(job.SITES)
    if not empty_sites:
        assert actual["rows"][0] == ("east", 3, 13000, 13000 / 3)
        assert actual["rows"][1] == ("north", 1, None, None)


def test_missing_database_is_not_created(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(job, "collab", SimpleNamespace(get_app_prop=lambda name: str(path)))
    with pytest.raises(sqlite3.OperationalError):
        job.query()
    assert not path.exists()


@pytest.mark.parametrize("failed", [False, True])
def test_missing_or_failed_site_rejects_partial_result(tmp_path, monkeypatch, failed):
    responses = FrozenResultQueue(
        [(site, []) for site in job.SITES[:-1]],
        {"site-3": RuntimeError("Client failed")} if failed else {},
        2,
    )
    group = Mock()
    group.query.return_value = responses
    monkeypatch.setattr(job, "collab", SimpleNamespace(get_clients=Mock(return_value=group)))
    with pytest.raises(RuntimeError, match="Every site"):
        job.run()
    assert not (tmp_path / "result.json").exists()
