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
import sqlite3
from contextlib import closing
from pathlib import Path

# Unequal group sizes and NULL amounts make incorrect AVG-of-AVGs easy to catch.
SITE_ROWS = {
    "site-1": [(1, "east", 1000), (2, "east", 3000), (3, "west", 7000)],
    "site-2": [(1, "east", 9000), (2, "west", None)],
    "site-3": [(1, "west", 1000), (2, "west", 4000), (3, "north", None)],
}
SCHEMA = "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT NOT NULL, amount_cents INTEGER)"


def prepare_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for site, rows in SITE_ROWS.items():
        path = root / f"{site}.db"
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}; use an empty directory")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(SCHEMA)
            connection.executemany("INSERT INTO sales VALUES (?, ?, ?)", rows)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create three example SQLite databases with identical schemas")
    parser.add_argument("--db-root", default="/tmp/nvflare/distributed-sql/data")
    prepare_data(parser.parse_args().db_root)
