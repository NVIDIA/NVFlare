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

The steps have separate responsibilities:

- `prepare_data.py`: create and populate the local Parquet files/SQL tables and write site bindings and policies before the federated job.
- `job.py`: configure and run `CatalogRecipe`, then print the published catalog.
- `discover.py`: inspect metadata and compare published table schemas.
- `verify_result.py`: optionally verify a completed run.

The job imports no preparation module and does not create or populate source tables.

## Install and run

From this directory in an activated Python environment:

```bash
python -m pip install -r requirements.txt
python prepare_data.py --site-1-format parquet --site-2-format parquet
python job.py
python discover.py
```

The default creates Parquet sources at both sites under
`/tmp/nvflare/federated-catalog/data`. Site-1 has 64 synthetic persons and site-2
has 96 in the main `omop` extract, with two visits and associated clinical events
per person. Each site also has smaller, independent department/registry fixtures.
These fixture sizes are known from preparation; the catalog does not calculate
population sizes or clinical cohort membership.

Each site has **three published datasets plus one withheld internal dataset**:

| Site | Dataset | Published tables | Main difference |
| --- | --- | ---: | --- |
| site-1 | `omop` | 6 | General clinical extract |
| site-1 | `cardiology` | 6 | Separate, smaller synthetic extract |
| site-1 | `oncology` | 5 | Site-specific dataset; no measurement table |
| site-2 | `omop` | 4 | Measurement table withheld by policy |
| site-2 | `cardiology` | 5 | Measurement table shared; no drug exposure table |
| site-2 | `registry` | 4 | Site-specific dataset; no measurement or drug exposure tables |

Dataset names label illustrative extracts, not validated disease cohorts. Each
has its own source binding, metadata, schema versions and publication rule.
Table names such as `person` belong to their containing dataset; they are not
merged across datasets or sites. Each fixture has a separate local ID range.

## Variation: Parquet and an RDBMS

Use SQLite for site-2 to demonstrate an actual relational database without an
external database service. `job.py` stays the same:

```bash
python prepare_data.py --site-1-format parquet --site-2-format sqlite
python job.py
python discover.py
```

The example includes the actual site configurations as readable files:

| File | Purpose |
| --- | --- |
| [config/site-1/catalog_site.json](config/site-1/catalog_site.json) | Explicit site-1 Parquet configuration |
| [config/site-2/catalog_site.json](config/site-2/catalog_site.json) | SQLite configuration for the mixed variation |
| [config/site-1/catalog_site.sqlite.json](config/site-1/catalog_site.sqlite.json) | Optional site-1 SQLite variation |
| [config/site-2/catalog_site.parquet.json](config/site-2/catalog_site.parquet.json) | Site-2 configuration for the Parquet-only variation |
| [config/site-1/default.json](config/site-1/default.json), [config/site-2/default.json](config/site-2/default.json) | Site-owned publication policies |
| [config/site-1/catalog_metadata.json](config/site-1/catalog_metadata.json), [config/site-2/catalog_metadata.json](config/site-2/catalog_metadata.json) | Optional OMOP declarations and Parquet key/relationship annotations |

Site-1's complete `catalog_site.json`:

```json
{"format_version": 1, "default_connector": "parquet"}
```

Site-2's complete mixed-source `catalog_site.json`:

```json
{"format_version": 1, "default_connector": "sqlite"}
```

Both sites have their own explicit configuration; neither site supplies defaults for the other. `default_connector` is a reader default within that site only.

Preparation **copies these checked-in files** into each site's `local/fdn/`;
it does not construct catalog configuration or policy JSON dynamically. It
creates/populates the source tables and writes `study_runtime.yaml` with the
absolute paths appropriate for `--data-root`. Those existing runtime bindings
already name each dataset, so catalog configuration need not repeat them when
they use the site's default connector.

A site with different formats within one Study can add only its exceptions:

```json
{
  "format_version": 1,
  "studies": {
    "default": {
      "registry": {"connector": "sqlite"}
    }
  }
}
```

The executor maps source names to built-in readers; no `catalog_reader=` argument
is needed. Built-in connectors accept no `options`, repeated `storage_format`,
class paths, or inline metadata. Optional declarations live beside the source:
`catalog.json` inside a Parquet dataset or `<database>.catalog.json` beside a
SQLite file (`omop.sqlite` uses `omop.catalog.json`). Preparation copies the
checked-in metadata file to those locations. SQLite observes keys/relationships
from the database; Parquet uses the site's optional annotations. Metadata fields
and an installed-operations list are not required for a basic catalog read. The
example advertises no execution operations.

The normative input contract is in the
[catalog design](../../../docs/design/federated_data_cataloging_design.md#catalog_sitejson-contract-v1).

The SQLite connector opens the database [read-only](https://www.sqlite.org/uri.html)
and inspects `sqlite_schema`, `PRAGMA table_info`, and `PRAGMA foreign_key_list`.
It does not select clinical rows. Parquet schema reads use file footers; keys and
relationships come from the site's sidecar declarations. SQLite keys and
relationships come from database constraints. Both produce `CatalogInfo` and pass
through the same site policy and validation. This example inspects ordinary tables;
remote PostgreSQL/MySQL credentials and database services are not required.

## Overlapping and different schemas

The main `omop` dataset illustrates the following table coverage. Other datasets
have their own coverage, as listed above.

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
python discover.py --dataset omop --table person

# Show every table, column/type/nullability, primary key and relationship.
python discover.py --dataset omop --show-schema
```

The default query reports a candidate per **site/dataset**, not just per site.
For example, site-2/omop lacks published `measurement`, but site-2/cardiology
has it. Requiring only the common clinical tables includes all six datasets. The script
also prints common columns, site-specific additions, and differing storage types.
A missing catalog table may reflect source coverage or publication policy; the
consumer cannot infer which from its absence. A metadata candidate is not proof
of patient eligibility, join correctness, clinical equivalence, or analysis access.
This example does not advertise or execute analysis operations.

The full result is printed by `job.py` and saved to
`/tmp/nvflare/federated-catalog/store/discovery.json`. A fresh store has six dataset
entries across the two sites, with the table counts listed above. Catalog/schema
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

Edit `/tmp/nvflare/federated-catalog/data/site-2/local/fdn/default.json` as
site-2's local owner. Set `datasets.omop.share` to `false` and keep the other
publication rules. The complete [withdrawal policy](config/site-2-withdraw-omop.json) is:

```json
{
  "format_version": 1,
  "datasets": {
    "omop": {
      "share": false,
      "exclude_columns": {
        "person": [
          "person_source_value",
          "month_of_birth",
          "day_of_birth"
        ],
        "visit_occurrence": [
          "visit_source_value"
        ],
        "condition_occurrence": [
          "condition_source_value"
        ],
        "measurement": [
          "measurement_source_value"
        ]
      },
      "exclude_metadata": [
        "owner"
      ],
      "exclude_tables": [
        "measurement"
      ]
    },
    "cardiology": {
      "share": true,
      "exclude_columns": {
        "person": [
          "person_source_value",
          "month_of_birth",
          "day_of_birth"
        ],
        "visit_occurrence": [
          "visit_source_value"
        ],
        "condition_occurrence": [
          "condition_source_value"
        ],
        "measurement": [
          "measurement_source_value"
        ]
      },
      "exclude_metadata": [
        "owner"
      ]
    },
    "registry": {
      "share": true,
      "exclude_columns": {
        "person": [
          "person_source_value",
          "month_of_birth",
          "day_of_birth"
        ],
        "visit_occurrence": [
          "visit_source_value"
        ],
        "condition_occurrence": [
          "condition_source_value"
        ]
      },
      "exclude_metadata": [
        "owner"
      ]
    },
    "internal": {
      "share": false
    }
  }
}
```

Save the updated JSON at the site policy path, then rerun:

```bash
python job.py
python verify_result.py
```

Discovery now contains five dataset entries: only site-2/omop is withdrawn,
while site-2/cardiology and site-2/registry remain visible. Previous names are
recovered from persistent state so a site can withdraw an individual dataset. Re-run preparation and the
job to restore initial policies and sharing.

## Customize paths

```bash
python prepare_data.py --data-root /tmp/my-catalog/data --site-1-format parquet --site-2-format sqlite
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
