# Hello PyTorch with MLFlow Experimental Tracking

This example, show that one can using the existing tracker code written for Tensorboard, 
but change the metrics tracking with both Tensorflow and MLFLow 

The only change compare with hello-pt-tb example is adding the following component in fed_server_config.json

```json
  {
      "id": "mlflow_receiver",
      "path": "nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLFlowReceiver",
      "args": {
        "kwargs": {"experiment_name": "hello-pt-experiments"},
        "artifact_location": "artifacts"
      }
  }

```