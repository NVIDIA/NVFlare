# Federated Learning on OMOP CDM Data

This tutorial runs one SQL study at three hospitals that store their records in the OMOP Common Data Model (CDM) and fits a GI bleeding risk model across them with NVFlare.
No patient rows leave a site, yet the federated model is identical to the one you would get by pooling all patients.

<img src="./figs/omop_federated.svg" alt="Sites map their local codes to standard OMOP concepts, run the same cohort SQL, and share only gradients and Hessians with the NVFlare server" width="900"/>

## What is OMOP?

The [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/), maintained by [OHDSI](https://www.ohdsi.org/), gives every site the same tables and maps every local code to a standard concept, so an ICD-10-CM `K92.2` and a SNOMED `74474003` diagnosis both become concept `192671` (Gastrointestinal hemorrhage).
A study written once against the CDM therefore runs unchanged at every site and yields the same features in the same order, which is exactly what federated learning needs.

## The study

The data is [Eunomia GiBleed](https://github.com/OHDSI/EunomiaDatasets), an OHDSI teaching dataset of 2,694 synthetic patients (Apache 2.0), split at random into three sites holding 55%, 30%, and 15% of the patients.
[cohort.py](./cohort.py) selects patients starting celecoxib or diclofenac, labels whether they had a GI hemorrhage within a year, and uses age, sex, the NSAID started, and prior GI conditions and painkiller use as covariates.
NVFlare ships `cohort.py` with [client.py](./client.py) to every site, each site sends back the gradient and Hessian of a logistic regression, and `FedAvgLrRecipe` takes a Newton-Raphson step with their average.

## Run it

```bash
pip install -r requirements.txt
python prepare_data.py   # download Eunomia and write one OMOP CDM per site
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

## Use your own OMOP CDM

Change the study by editing the concept IDs in `cohort.py`, which you can look up in [Athena](https://athena.ohdsi.org/).
Point `load_cohort()` at your own database by changing its `duckdb.connect` call, for example to a PostgreSQL CDM through DuckDB's [postgres extension](https://duckdb.org/docs/stable/core_extensions/postgres).
`python job.py --export --export-dir /tmp/omop_job` writes a job you can submit to a real NVFlare deployment, and `python -m pytest tests` runs the example's tests.
