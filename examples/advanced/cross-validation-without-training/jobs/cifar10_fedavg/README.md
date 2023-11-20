# Run cross site validation using the previous trained results

## Introduction

This example shows how to run the NVFlare cross-site validation without the training workflow, making use of the previous run results. 

### Previous run best global model and local best model

The previous run best global model and best local model are stored in the "server" and "client" sub-folder separately under the "models" folder. 

### How to run the Job

Define two OS system variable "SERVER_MODEL_DIR" and "CLIENT_MODEL_DIR" to point to the absolute path of the server best model and local best model location respectively. Then use the NVFlare admin command "submit_job" to submit and run the cross-validation job.

