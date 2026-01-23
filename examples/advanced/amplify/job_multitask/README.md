# AMPLIFY Multi-Task Federated Fine-tuning

This directory contains the job configuration for federated multi-task fine-tuning of AMPLIFY, where each client trains on a different downstream task while jointly fine-tuning the shared AMPLIFY trunk.

## Overview

In this scenario:
- **6 clients** (one per task): aggregation, binding, expression, immunogenicity, polyreactivity, thermostability
- Each client trains their **own private regression head** for their specific task
- All clients **jointly fine-tune the AMPLIFY trunk** using FedAvg
- Regression heads are kept private (not shared with server) using the `ExcludeParamsFilter`

## Key Components

- **job.py**: Main job configuration using `FedAvgRecipe`
- **../client.py**: Shared client training script (used by both scenarios)
- **../src/model.py**: AmplifyRegressor model definition
- **../src/filters.py**: ExcludeParamsFilter to keep regressors private
