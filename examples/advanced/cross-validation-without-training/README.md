# Cifar10_fedavg example re-run cross site validation using the previous trained results

## Introduction

This example shows how to re-run the NVFlare cross-site validation without the training workflow, making use of the previous run results. The example uses the cifar10_fedavg NVFlare job configuration.

## Instructions

The previous run server global models and client local models are stored in the "models" folder. 

run the ./setup.sh to set up the proper config_fed_client.json and environment.json. Then you can run this job for cross-site validation without training workflow.