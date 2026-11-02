# Federated Learning on OMOP CDM Data

This tutorial connects NVFlare to hospitals that already store their records in an OMOP Common Data Model (CDM) database.
Every site runs the same cohort query against its own tables, turns the result into features, and trains a shared GI bleeding risk model, while patient rows never leave the site.

<img src="./figs/omop_federated.svg" alt="Each site runs cohort.py against its own OMOP CDM, where different local codes resolve to the same standard concept, and client.py shares only gradients and Hessians with the NVFlare server" width="900"/>

## What is OMOP?

The [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/), maintained by [OHDSI](https://www.ohdsi.org/), gives every site the same tables and maps every local code to a standard concept, so an ICD-10-CM `K92.2` and a SNOMED `74474003` diagnosis both become concept `192671` (Gastrointestinal hemorrhage).
A study written once against the CDM therefore runs unchanged at every site and yields the same features in the same order, which is exactly what federated learning needs.

## What each site runs

The study is defined once in [cohort.py](./cohort.py): patients starting celecoxib or diclofenac, whether they had a GI hemorrhage within a year, and their age, sex, NSAID, prior GI conditions and painkiller use as covariates.
Its SQL finds conditions and drugs through the standard vocabulary instead of local codes, which is what lets one query serve every site:

```sql
select co.person_id, co.condition_start_date
from condition_occurrence co
join concept_ancestor ca on ca.descendant_concept_id = co.condition_concept_id
where ca.ancestor_concept_id = 192671  -- Gastrointestinal hemorrhage and every more specific concept
```

NVFlare ships `cohort.py` with [client.py](./client.py) to every site.
`client.py` runs the query with `load_cohort()`, computes the gradient and Hessian of a logistic regression on the resulting features, and sends only those to the server, where `FedAvgLrRecipe` takes a Newton-Raphson step with their average.

## Connect your OMOP CDM

`--cdm` points each site at its CDM, either a DuckDB file or a PostgreSQL connection string read through DuckDB's [postgres extension](https://duckdb.org/docs/stable/core_extensions/postgres), and `--cdm_schema` names the schema holding the CDM tables.
The job can refer to the connection string as an NVFlare secret, so each site's credentials stay in that site's environment:

```bash
python job.py --cdm '${secret:OMOP_CDM}' --cdm_schema cdm --export --export-dir /tmp/omop_job
```

Each site sets the variable before starting its NVFlare client, for example `export OMOP_CDM=postgres:postgresql://flare:password@omop-db:5432/omop`, and you submit `/tmp/omop_job` to your deployment.
To study something else, edit the concept IDs in `cohort.py`, which you can look up in [Athena](https://athena.ohdsi.org/).

## Try it with simulated sites

Without a hospital network at hand, `prepare_data.py` stands in for one.
It downloads [Eunomia GiBleed](https://github.com/OHDSI/EunomiaDatasets), an OHDSI teaching dataset of 2,694 synthetic patients (Apache 2.0), and splits it at random into one DuckDB CDM per site holding 55%, 30%, and 15% of the patients.
By default `job.py` sets `--cdm` to `/tmp/nvflare/dataset/omop/{SITE_NAME}.duckdb`, and NVFlare fills in each site's name.

```bash
pip install -r requirements.txt
python prepare_data.py   # simulation setup: one DuckDB OMOP CDM per site
python job.py            # run the federated study in the NVFlare simulator
python evaluate.py       # compare with a pooled fit and with each site alone
```

```
Largest difference between federated and pooled coefficients: 3.4e-06

Odds ratio (95% CI)                  federated        site-1 alone        site-2 alone        site-3 alone
age_decades                   1.31 (0.95-1.80)    1.43 (0.93-2.21)    1.32 (0.74-2.35)    0.89 (0.37-2.15)
male                          1.00 (0.81-1.24)    0.88 (0.66-1.17)    1.34 (0.90-1.98)    1.01 (0.57-1.78)
celecoxib                     0.96 (0.75-1.23)    1.29 (0.91-1.81)    0.77 (0.49-1.20)    0.58 (0.31-1.07)
peptic_ulcer                  4.61 (3.09-6.87)    3.99 (2.32-6.88)   7.01 (3.25-15.13)   4.16 (1.49-11.65)
...
```

The federated coefficients match a fit on all patients pooled, which a real network could never compute.
Each site alone gets far wider confidence intervals, for example 1.49 to 11.65 for prior peptic ulcer at site 3 against 3.09 to 6.87 when federated.
`python -m pytest tests` runs the example's tests.
