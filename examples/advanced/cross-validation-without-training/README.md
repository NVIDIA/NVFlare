# Cifar10_fedavg example re-run cross site validation using the previous trained results

## Introduction

This example shows how to re-run the NVFlare cross-site validation without the training workflow, making use of the previous run results. The example uses the cifar10_fedavg NVFlare job configuration.

### How the example been built

This example uses the exact same NVFlare job definition of cifar10_fedavg, which contains the training workflow, followed by the cross-site validation workflow. These are the steps of changes made to enable re-run the cross-site validation without the training.

1. removed the scatter_gather_ctl workflow from the config_fed_server.json

2.  In the PTFileModelLocator, add the following lines to initialize and create the persistence_manager instance in the model_persister. This is a NVFlare core codes change.

        if not self.model_persistor.persistence_manager:
            self.model_persistor.load_model(fl_ctx)
3. In order to locate the global models, create an config/environment.json file, which contains the following lines to indicate where the global model and best global model exist on the server.
```
    {
      "APP_CKPT_DIR": "$SERVER_MODEL_DIR"
    }
```

4. In order to allow the client to locate the local model and local best model for cross-validation, modify the CIFAR10ModelLearner to add a "model_dir" optional argument. When this "model_dir" is provided, CIFAR10ModelLearner will locate the local models in this folder.

5. Change the config_fed_client.json to include the "model_dir" configuration.


## Instructions

The previous run server global models and client local models are stored in the "models" folder. 

run the ./setup.sh to set up the proper config_fed_client.json and environment.json. Then you can run this job for cross-site validation without training workflow.