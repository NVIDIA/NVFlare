# Federated Catalog: heterogeneous OMOP sources

Discover clinical dataset metadata across two simulated sites using `CatalogRecipe`
and `SimEnv`. The example follows the federated statistics/Hello PyTorch structure:
prepare data, run a short `job.py`, then inspect or verify results separately.
It runs on CPU and downloads no datasets.

The fixture is a deterministic, **synthetic OMOP CDM 5.4-shaped subset** based on
[OHDSI's CDM specification](https://ohdsi.github.io/CommonDataModel/cdm54.html).
It contains selected columns and tables, not a complete CDM database. Concept IDs
are 0 (unmapped), no vocabulary release is installed, and no clinical or vocabulary
validation is claimed. The catalog reports site declarations and observed schemas.

## Install and run

From this directory in an activated Python environment:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python job.py
python discover.py
```

The default creates Parquet sources at both sites under
`/tmp/nvflare/federated-catalog/data`. Site-1 has 64 synthetic persons and site-2
has 96, with two visits and associated clinical events per person. These fixture
sizes are known from preparation; the catalog does not calculate population sizes.

## Variation: Parquet and an RDBMS

Use SQLite for site-2 to demonstrate an actual relational database without an
external database service. `job.py` stays the same:

```bash
python prepare_data.py --site-2-format sqlite
python job.py
python discover.py
```

The site-owned `local/study_runtime.yaml` binds `omop` to either its Parquet
directory or SQLite database. Its `local/fdn/catalog_site.json` selects a built-in
connector per dataset:

```json
{
  "format_version": 1,
  "studies": {
    "default": {
      "omop": {"connector": "sqlite"}
    }
  }
}
```

Use `"parquet"` at site-1. Preparation also adds site-declared profile metadata in
binding `options`; the Parquet version supplies declarations in `catalog.json`.
The executor maps connector names to their reader implementations. A global recipe
format would not represent these different site sources, so the example needs no
`catalog_reader=` argument. That argument remains available for custom readers.

The SQLite connector opens the database [read-only](https://www.sqlite.org/uri.html)
and inspects `sqlite_schema`, `PRAGMA table_info`, and `PRAGMA foreign_key_list`.
It does not select clinical rows. Parquet schema reads use file footers; keys and
relationships come from the site's sidecar declarations. SQLite keys and
relationships come from database constraints. Both produce `CatalogInfo` and pass
through the same site policy and validation. This example inspects ordinary tables;
remote PostgreSQL/MySQL credentials and database services are not required.

## Overlapping and different schemas

| Table | Site-1 source | Site-2 source | Published view |
| --- | --- | --- | --- |
| `person` | Present | Present | Source identifiers and birth month/day fields excluded |
| `observation_period` | Present | Present | Published at both sites |
| `visit_occurrence` | Present | Present | Source-value column excluded |
| `condition_occurrence` | Present | Present | Common core plus site-specific columns |
| `drug_exposure` | Present | Absent | Published only by site-1 |
| `measurement` | Present | Present | Withheld by site-2's policy |

The common `condition_occurrence` columns are `condition_occurrence_id`,
`person_id`, `condition_concept_id`, `condition_start_date`, and
`condition_type_concept_id`. Site-1 additionally has `condition_end_date` and
`visit_occurrence_id`; site-2 has `condition_status_concept_id` instead. The
catalog preserves these differences rather than filling missing columns or
assuming equivalent schemas.

In mixed storage, physical types also differ: Parquet dates are `date32[day]`,
while the SQLite fixture stores dates as ISO text. `discover.py` reports observed
types without asserting semantic equivalence or performing conversions.

Each site also has an `internal` identity-crosswalk dataset which is never shared.
All `*_source_value` columns and owner metadata are excluded from the published
OMOP view. Schema column names such as `person_id` remain to describe joins;
actual identifier values and clinical rows remain local.

## Discover candidate sites

```bash
# Default: require person, condition_occurrence and measurement tables.
python discover.py

# Require only the common clinical tables.
python discover.py --require-tables person condition_occurrence

# Compare overlapping and site-specific columns of another table.
python discover.py --table person
```

The first query identifies site-1 as a metadata candidate and reports missing
`measurement` metadata at site-2. The second identifies both sites. The script
also prints common columns, site-specific additions, and differing storage types.
A missing catalog table may reflect source coverage or publication policy; the
consumer cannot infer which from its absence. A metadata candidate is not proof
of patient eligibility, join correctness, clinical equivalence, or analysis access.
The published `statistics` operation is a declaration; this example does not run
federated statistics or a cohort query.

The full result is printed by `job.py` and saved to
`/tmp/nvflare/federated-catalog/store/discovery.json`. A fresh store has two `omop`
entries with six published tables at site-1 and four at site-2. Catalog/schema
versions start at 1; unchanged reads retain their versions. Simulator logs and
intermediate manifests are under
`/tmp/nvflare/federated-catalog/simulation/catalog_read/`.

## Optional verification

```bash
python verify_result.py
```

This checks the saved output's table/column differences, declared profile,
relationships, backend provenance and policy exclusions. Verification is separate
from `job.py`. Run it after rerunning the job whenever a policy changes.

## Withdraw one site's metadata

Acting as site-2's local owner, change its policy:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('/tmp/nvflare/federated-catalog/data/site-2/local/fdn/default.json')
policy = json.loads(path.read_text())
policy['datasets']['omop']['share'] = False
path.write_text(json.dumps(policy, indent=2) + '\n')
PY
python job.py
python verify_result.py
```

Discovery now contains only site-1. Previous names are recovered from persistent
state so site-2 can withdraw its earlier publication. Re-run preparation and the
job to restore initial policies and sharing.

## Customize paths

```bash
python prepare_data.py --data-root /tmp/my-catalog/data --site-2-format sqlite
python job.py --data-root /tmp/my-catalog/data \
  --workspace /tmp/my-catalog/simulation --store-root /tmp/my-catalog/store
python discover.py --store-root /tmp/my-catalog/store
python verify_result.py --data-root /tmp/my-catalog/data --store-root /tmp/my-catalog/store
```

Use a dedicated simulation workspace: `SimEnv` replaces the job workspace on each
run. Data and persistent state live outside it. `NVFLARE_SIMULATOR_WORKSPACE_ROOT`,
when set, overrides `--workspace`.

## Recipe and deployment

`job.py` only configures `CatalogRecipe`, executes it with `SimEnv`, and prints
results. The recipe owns dispatch, collection, reconciliation and persistence.
`site_config_root` locates the site-owned directories on this shared host and is
allowed only for simulation. There are no example controllers or executors.

In a provisioned federation, omit `site_config_root` and configure the server's
`CatalogManager` and each site's policies/runtime bindings. The parent manager
provides authenticated private handoff and archive/download redaction. Simulation
keeps unsigned manifests in its local workspace; it does not exercise admin login,
scheduling or those production artifact protections.
