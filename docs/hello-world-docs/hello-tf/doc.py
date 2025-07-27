
"""
`hello pytorch <../hello-pt/doc.html>`_ ||
`hello lightning <../hello-lightning/doc.html>`_ ||
**hello tensorflow** ||
`hello LR <../hello-lr/doc.html>`_ ||
`hello KMeans <../hello-kmeans/doc.html>`_ ||
`hello KM <../hello-km/doc.html>`_ ||
`hello stats <../hello-stats/doc.html>`_ ||
`hello cyclic <../hello-cyclic/doc.html>`_ ||
`hello-xgboost <../hello-xgboost/doc.html>`_ ||
`hello-flower <../hello-flower/doc.html>`_ ||


Hello Tensorflow
===================

This example demonstrates how to use NVIDIA FLARE with **Tensorflow** to train an image classifier using
cyclic weight transfer approach.The complete example code can be found in the`hello-tf directory <examples/hello-world/hello-tf/>`_.
It is recommended to create a virtual environment and run everything within a virtualenv.

NVIDIA FLARE Installation
-------------------------
for the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare

Install the dependency

.. code-block:: text

    pip install -r requirements.txt


Code Structure
--------------

first get the example code from github:

.. code-block:: text

    git clone https://github.com/NVIDIA/NVFlare.git

then navigate to the hello-pt directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-tf


.. code-block:: text

    hello-pt
        |
        |-- client.py         # client local training script
        |-- model.py          # model definition
        |-- job.py            # job recipe that defines client and server configurations
        |-- requirements.txt  # dependencies

Data
-----------------
This example uses the `MNIST dataset`

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = (
        train_images / 255.0,
        test_images / 255.0,
    )

    # simulate separate datasets for each client by dividing MNIST dataset in half
    client_name = flare.get_site_name()
    if client_name == "site-1":
        train_images = train_images[: len(train_images) // 2]
        train_labels = train_labels[: len(train_labels) // 2]
        test_images = test_images[: len(test_images) // 2]
        test_labels = test_labels[: len(test_labels) // 2]
    elif client_name == "site-2":
        train_images = train_images[len(train_images) // 2 :]
        train_labels = train_labels[len(train_labels) // 2 :]
        test_images = test_images[len(test_images) // 2 :]
        test_labels = test_labels[len(test_labels) // 2 :]




"""
################################
# Model
# ------------------
# we are leveraging the Keras API to define the model.
from tensorflow.keras import layers, models

class TFNet(models.Sequential):
    def __init__(self, input_shape=(None, 28, 28)):
        super().__init__()
        self._input_shape = input_shape
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation="relu"))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(10))


######################################################################
# --------------
#


#####################################################################
# Client Code
# ------------------
#
# Notice the training code is almost identical to the pytorch standard training code.
# The only difference is that we added a few lines to receive and send data to the server.
#

import tensorflow as tf

import nvflare.client as flare

WEIGHTS_PATH = "./tf_model.weights.h5"


def main():
    flare.init()

    model = TFNet()
    model.build(input_shape=(None, 28, 28))
    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = (
        train_images / 255.0,
        test_images / 255.0,
    )

    # simulate separate datasets for each client by dividing MNIST dataset in half
    client_name = flare.get_site_name()
    if client_name == "site-1":
        train_images = train_images[: len(train_images) // 2]
        train_labels = train_labels[: len(train_labels) // 2]
        test_images = test_images[: len(test_images) // 2]
        test_labels = test_labels[: len(test_labels) // 2]
    elif client_name == "site-2":
        train_images = train_images[len(train_images) // 2 :]
        train_labels = train_labels[len(train_labels) // 2 :]
        test_images = test_images[len(test_images) // 2 :]
        test_labels = test_labels[len(test_labels) // 2 :]

    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}")

        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        _, test_global_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the test images: {test_global_acc * 100} %"
        )

        # training
        model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

        print("Finished Training")

        model.save_weights(WEIGHTS_PATH)

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}", flush=True)
        print(f"finished round: {input_model.current_round}", flush=True)

        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers},
            params_type="FULL",
            metrics={"accuracy": test_global_acc},
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()


#####################################################################
# Server Code
# ------------------
# In federated averaging, the server code is responsible for
# aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
# In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
# The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
# There is no need to defined a customized server code for this example.


#####################################################################
# Job Recipe Code
# ------------------
# Job Recipe contains the client.py and built-in fedavg algorithm.

from model import TFNet
from nvflare.job_config.Job_recipe import FedAvgRecipe

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "client.py"
    client_script_args = ""

    recipe = FedAvgRecipe(name="hello-tf",
                          min_clients=n_clients,
                          num_rounds=num_rounds,
                          model=TFNet(),
                          client_script=train_script,
                          client_script_args=client_script_args)

    recipe.execute(clients=n_clients, gpus=0)  # defaul to SimEnv




#####################################################################
# Run FL Job
# ------------------
#
# This section provides the command to execute the federated learning job
# using the job recipe defined above. Run this command in your terminal.


#####################################################################
# **Command to execute the FL job**
#
# Use the following command in your terminal to start the job with the specified
# number of rounds, batch size, and number of clients.
#
#
# .. code-block:: text
#
#   python job.py --num_rounds 2 --batch_size 16


#####################################################################
# output
#
# .. code-block:: text
#
#