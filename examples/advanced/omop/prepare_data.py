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

"""Download the OHDSI Eunomia GiBleed OMOP CDM and split its patients into one DuckDB database per site."""

import argparse
import hashlib
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np

EUNOMIA_URL = (
    "https://raw.githubusercontent.com/OHDSI/EunomiaDatasets/"
    "a11ad148a6b1b1fafb75a1e40c504c887db76fa4/datasets/GiBleed/GiBleed_5.3.zip"
)
EUNOMIA_SHA256 = "081b82e84c90e2904e02c26546e46412b01cce566071c77e0c4a355c2d0612f9"
DATA_ROOT = "/tmp/nvflare/dataset/omop"


def download_cdm(data_root: Path) -> Path:
    """Download and extract the Eunomia GiBleed CSV files.

    Args:
        data_root: Directory that receives the archive and the extracted CSV files.

    Returns:
        The directory holding one CSV file per OMOP CDM table.
    """
    archive = data_root / "GiBleed_5.3.zip"
    if not archive.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(EUNOMIA_URL) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EUNOMIA_SHA256:
        archive.unlink()
        raise ValueError(f"checksum mismatch for {EUNOMIA_URL}, removed the download")
    with zipfile.ZipFile(archive) as files:
        files.extractall(data_root, [name for name in files.namelist() if name.startswith("GiBleed_5.3/")])
    return data_root / "GiBleed_5.3"


def split_sites(cdm_dir: Path, data_root: Path, fractions: Sequence[float], seed: int) -> list[Path]:
    """Assign every patient to one site and write each site's OMOP CDM.

    Clinical tables keep only the site's own patients, while vocabulary and metadata tables are copied whole.

    Args:
        cdm_dir: Directory holding one CSV file per OMOP CDM table.
        data_root: Directory that receives one ``site-<n>.duckdb`` database per site.
        fractions: Share of the patients assigned to each site.
        seed: Seed for the random assignment.

    Returns:
        The site databases in site order.

    Raises:
        ValueError: If a fraction is not positive or the fractions do not sum to 1.
    """
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must be positive and sum to 1, got {list(fractions)}")
    tables = {path.stem.lower(): path for path in sorted(cdm_dir.glob("*.csv"))}
    person_ids = duckdb.read_csv(str(tables["person"]), normalize_names=True).fetchnumpy()["person_id"]
    shuffled = np.random.default_rng(seed).permutation(np.sort(person_ids))
    cuts = (np.cumsum(fractions)[:-1] * len(shuffled)).astype(int)
    databases = []
    for number, members in enumerate(np.split(shuffled, cuts), start=1):
        database = data_root / f"site-{number}.duckdb"
        database.unlink(missing_ok=True)
        with duckdb.connect(str(database)) as site:
            site.execute("create temp table members as select unnest(?::bigint[]) as person_id", [members.tolist()])
            for table, path in tables.items():
                rows = site.read_csv(str(path), normalize_names=True)
                if "person_id" in rows.columns:
                    rows = rows.filter("cast(person_id as bigint) in (from members)")
                rows.create(table)
        print(f"site-{number}: {len(members)} patients -> {database}")
        databases.append(database)
    return databases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.55, 0.3, 0.15])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    split_sites(download_cdm(data_root), data_root, args.fractions, args.seed)
