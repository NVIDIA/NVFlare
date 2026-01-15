.. _client_api_usage:

################
Client API Usage
################

The FLARE Client API provides an easy way for users to convert their centralized,
local training code into federated learning code with the following benefits:

* Only requires a few lines of code changes, without the need to restructure the code or implement a new class
* Reduces the number of new FLARE specific concepts exposed to users
* Easy adaptation from existing local training code using different frameworks
  (PyTorch, PyTorch Lightning, HuggingFace)

Core concept
============

The general structure of the popular federated learning (FL) workflow, "FedAvg" is as follows:

#. FL server initializes an initial model
#. For each round (global iteration):

   #. FL server sends the global model to clients
   #. Each FL client starts with this global model and trains on their own data
   #. Each FL client sends back their trained model
   #. FL server aggregates all the models and produces a new global model

On the client side, the training workflow is as follows:

#. Receive the model from the FL server
#. Perform local training on the received global model and/or evaluate the
   received global model for model selection
#. Send the new model back to the FL server

To convert a centralized training code to federated learning, we need to
adapt the code to do the following steps:

#. Obtain the required information from received :ref:`fl_model`
#. Run local training
#. Put the results in a new :ref:`fl_model` to be sent back

For a general use case, there are three essential methods for the Client API:

* ``init()``: Initializes NVFlare Client API environment.
* ``receive()``: Receives model from NVFlare side.
* ``send()``: Sends the model to NVFlare side.

Users can use the Client API to change their centralized training code to
federated learning, for example:

.. code-block:: python

    import nvflare.client as flare

    flare.init() # 1. Initializes NVFlare Client API environment.
    input_model = flare.receive() # 2. Receives model from NVFlare side.
    params = input_model.params # 3. Obtain the required information from received FLModel

    # original local training code begins
    new_params = local_train(params)
    # original local training code ends

    output_model = flare.FLModel(params=new_params) # 4. Put the results in a new FLModel
    flare.send(output_model) # 5. Sends the model to NVFlare side.

With 5 lines of code changes, we convert the centralized training code to
a federated learning setting.

After this, we can utilize the :ref:`job_recipe` to define and run the federated learning job.

Combining Client API with Job Recipe
=====================================

The Client API handles the **client-side training code** (what each client does with the model),
while the Job Recipe handles the **job definition** (how the FL workflow is configured and executed).

Here's how they work together in a complete example:

**File structure:**

.. code-block:: none

    my-fl-project/
    ├── job.py              # Job definition using Recipe API
    ├── client.py           # Client training script using Client API
    └── requirements.txt    # Dependencies

**client.py** - Client-side training using Client API:

.. code-block:: python

    import nvflare.client as flare
    import numpy as np

    def train(input_arr):
        # Simulate training by adding 1 to each element
        output_arr = input_arr + 1
        return output_arr

    def main():
        flare.init()

        while flare.is_running():
            # Receive global model from server
            input_model = flare.receive()
            params = input_model.params

            # Perform local training
            new_params = train(params)

            # Send updated model back to server
            output_model = flare.FLModel(params=new_params)
            flare.send(output_model)

    if __name__ == "__main__":
        main()

**job.py** - Job definition using Recipe API:

.. code-block:: python

    from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
    from nvflare.recipe import SimEnv

    # Define the federated learning job
    recipe = NumpyFedAvgRecipe(
        name="my-fl-job",
        min_clients=2,
        num_rounds=3,
        initial_model=[[1, 2, 3], [4, 5, 6]],
        train_script="client.py",
    )

    # Run in simulation environment
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)
    print(f"Job completed: {run.get_status()}")

**Running the job:**

.. code-block:: bash

    python job.py

This will run the federated learning job in simulation mode. The same recipe can be used
with different environments (``SimEnv``, ``PocEnv``, ``ProdEnv``) without changing the code.

For more details on Job Recipes, see :ref:`job_recipe`.

Below is a table overview of key Client APIs.

.. list-table:: Client API
   :widths: 25 25 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - init
     - Initializes NVFlare Client API environment.
     - :func:`init<nvflare.client.api.init>`
   * - receive
     - Receives model from NVFlare side.
     - :func:`receive<nvflare.client.api.receive>`
   * - send
     - Sends the model to NVFlare side.
     - :func:`send<nvflare.client.api.send>`
   * - system_info
     - Gets NVFlare system information.
     - :func:`system_info<nvflare.client.api.system_info>`
   * - get_job_id
     - Gets job id.
     - :func:`get_job_id<nvflare.client.api.get_job_id>`
   * - get_site_name
     - Gets site name.
     - :func:`get_site_name<nvflare.client.api.get_site_name>`
   * - is_running
     - Returns whether the NVFlare system is up and running.
     - :func:`is_running<nvflare.client.api.is_running>`
   * - is_train
     - Returns whether the current task is a training task.
     - :func:`is_train<nvflare.client.api.is_train>`
   * - is_evaluate
     - Returns whether the current task is an evaluation task.
     - :func:`is_evaluate<nvflare.client.api.is_evaluate>`
   * - is_submit_model
     - Returns whether the current task is a submit_model task.
     - :func:`is_submit_model<nvflare.client.api.is_submit_model>`

.. list-table:: Lightning APIs
   :widths: 25 25 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - patch
     - Patches the PyTorch Lightning Trainer for usage with FLARE.
     - :func:`patch<nvflare.app_opt.lightning.api.patch>`

.. list-table:: Metrics Logger
   :widths: 25 25 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - SummaryWriter
     - SummaryWriter mimics the usage of Tensorboard's SummaryWriter.
     - :class:`SummaryWriter<nvflare.client.tracking.SummaryWriter>`
   * - WandBWriter
     - WandBWriter mimics the usage of weights and biases.
     - :class:`WandBWriter<nvflare.client.tracking.WandBWriter>`
   * - MLflowWriter
     - MLflowWriter mimics the usage of MLflow.
     - :class:`MLflowWriter<nvflare.client.tracking.MLflowWriter>`


Framework-Specific Examples
===========================

The Client API works with various ML frameworks. Here are complete examples:

**PyTorch Example:**

.. code-block:: python

    # job.py
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import SimEnv
    from model import SimpleNetwork

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=2,
        num_rounds=2,
        initial_model=SimpleNetwork(),
        train_script="client.py",
        train_args="--batch_size 32",
    )

    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

.. code-block:: python

    # client.py
    import nvflare.client as flare

    def main():
        flare.init()

        while flare.is_running():
            input_model = flare.receive()

            # Load model weights
            net.load_state_dict(input_model.params)

            # Train locally
            train(net, train_loader)

            # Send back updated model
            output_model = flare.FLModel(
                params=net.state_dict(),
                metrics={"accuracy": accuracy}
            )
            flare.send(output_model)

For complete working examples, see:

* PyTorch: :github_nvflare_link:`hello-pt <examples/hello-world/hello-pt>`
* NumPy: :github_nvflare_link:`hello-numpy <examples/hello-world/hello-numpy>`
* PyTorch Lightning: :github_nvflare_link:`hello-lightning <examples/hello-world/hello-lightning>`
* TensorFlow: :github_nvflare_link:`hello-tf <examples/hello-world/hello-tf>`

Additional Resources
====================

For more details on Client API:

* Client API Module: :mod:`nvflare.client.api` - In-depth API documentation
* Communication Configuration: :ref:`client_api` - Advanced configuration details
* PyTorch Lightning API: :mod:`nvflare.app_opt.lightning.api` - Lightning-specific integration
* Job Recipe Guide: :ref:`job_recipe` - How to define and run FL jobs
