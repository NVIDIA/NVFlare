# Hello Numpy Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset. It uses the [CrossSiteModelEval](https://nvidia.github.io/NVFlare/apidocs/nvflare.app_common.workflows.html#nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval) controller workflow.

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
set_run_number 1
upload_app hello-numpy-cross-val
deploy_app hello-numpy-cross-val
start_app all
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
