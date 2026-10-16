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

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import PocEnv, SimEnv

SITES = ["site-1", "site-2", "site-3"]
LOCAL_SQL = """SELECT region, COUNT(*), COUNT(amount_cents), SUM(amount_cents)
               FROM sales GROUP BY region"""
# AVG needs the count of non-NULL amounts, which can differ from the row count.
MERGE_SQL = """SELECT region, SUM(row_count) AS sale_count, SUM(amount_sum) AS total_cents,
                      1.0 * SUM(amount_sum) / NULLIF(SUM(amount_count), 0) AS average_cents
               FROM partials GROUP BY region ORDER BY region"""


@collab.publish
def query():
    uri = Path(collab.get_app_prop("db_path")).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return db.execute(LOCAL_SQL).fetchall()


@collab.main
def run():
    responses = collab.get_clients(SITES).query()
    partials = dict(responses)
    if responses.failures or set(partials) != set(SITES):
        raise RuntimeError("Every site must return a successful SQL result")
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE partials (region, row_count, amount_count, amount_sum)")
        for rows in partials.values():
            db.executemany("INSERT INTO partials VALUES (?, ?, ?, ?)", rows)
        cursor = db.execute(MERGE_SQL)
        result = {"columns": [c[0] for c in cursor.description], "rows": cursor.fetchall(), "sites": SITES}
    output = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())) / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Distributed SQL example")
    parser.add_argument("--db-root", default="/tmp/nvflare/distributed-sql/data")
    parser.add_argument("--env", choices=("sim", "poc"), default="sim")
    args = parser.parse_args()

    # Uses this module's @collab.main and @collab.publish functions.
    recipe = CollabRecipe(job_name="distributed_sql", min_clients=len(SITES))
    recipe.set_per_site_config({site: {"db_path": str(Path(args.db_root).resolve() / f"{site}.db")} for site in SITES})
    env = PocEnv(clients=SITES, num_clients=len(SITES)) if args.env == "poc" else SimEnv(clients=SITES)
    execution = recipe.execute(env)
    result = execution.get_result(timeout=120, clean_up=False)
    if result is None or execution.get_status() != "FINISHED:COMPLETED":
        raise RuntimeError(f"SQL job failed: {execution.get_status()}")
    print("Results at:", result)


if __name__ == "__main__":
    main()
