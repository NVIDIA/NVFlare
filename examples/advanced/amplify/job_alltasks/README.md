# AMPLIFY All-Tasks Federated Fine-tuning

This directory contains the job configuration for federated all-tasks fine-tuning of AMPLIFY, where each client trains on all tasks using heterogeneous data distributions.

## Overview

In this scenario:
- **Multiple clients** (default: 6) each train on **all tasks**
- Data is split heterogeneously across clients using Dirichlet distribution
- Two modes available:
  1. **Shared regressors**: All regression heads are jointly trained and shared
  2. **Private regressors**: Regression heads remain private to each client (using `--private_regressors`)

## Key Components

- **job.py**: Main job configuration using `FedAvgRecipe`
- **../client.py**: Shared client training script (used by both scenarios)
- **../src/model.py**: AmplifyRegressor model definition
- **../src/filters.py**: ExcludeParamsFilter to keep regressors private (when `--private_regressors` is used)
