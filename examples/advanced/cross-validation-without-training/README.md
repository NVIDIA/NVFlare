# Cifar10_fedavg example re-run cross site validation using the previous trained results

## Introduction

This example shows how to re-run the NVFlare cross-site validation without the training workflow, making use of the previous run results. The example uses the cifar10_fedavg NVFlare job configuration.

### How is the example built

This example uses the exact same NVFlare job definition of cifar10_fedavg, which contains the training workflow, followed by the cross-site validation workflow. These are the steps of changes made to enable re-run the cross-site validation without the training.

1. removed the scatter_gather_ctl workflow from the config_fed_server.json

        self.model_persistor.load_model(fl_ctx)
2. Change the config_fed_server.json to add the "global_model_file_name" and "best_global_model_file_name" with the absolute paths to the global model and best global model locations. Also add  "model_dir" to the configuration. 
```
                "global_model_file_name": "{MODEL_DIR}/FL_global_model.pt",
                "best_global_model_file_name": "{MODEL_DIR}/best_FL_global_model.pt"
```

3. In order to allow the client to locate the local model and local best model for cross-validation, modify the CIFAR10ModelLearner to add a "model_dir" optional argument. When this "model_dir" is provided, CIFAR10ModelLearner will locate the local models in this folder.

4. Change the config_fed_client.json to include the "model_dir" configuration.


## Instructions

The previous run server global models and client local models are stored in the "models" folder. 

run the ./setup.sh to set up the proper config_fed_client.json and proper config_fed_server.json. Then you can run this job for cross-site validation without training workflow.